//! Unlimited-OCR's vision tower (spec 137 WS-03): an image's views (its 1024 global view and its 640 crops) to
//! the language model's image rows, `lm_features` of `UnlimitedOCRModel.forward` in the checkpoint's
//! `modeling_unlimitedocr.py`. Each view runs SAM (`vision_sam.zig`), CLIP on SAM's features and the projector
//! (`vision_clip.zig`); then the rows are laid out as the model reads them: the crops' rows as one grid (crop i
//! at row i / w, column i % w of `images_spatial_crop`'s [w, h]), each grid row followed by `image_newline`, then
//! the global view's rows likewise, then `view_seperator`.
//!
//! Views of one size run together, at most `crops_per_run` crops or `globals_per_run` global views a run; every
//! kernel keeps a row's bits its own (one GEMM tile whatever the rows, attention within a view, norms a row), so a
//! view's rows are the same bits alone or beside others.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const safetensors = @import("../../weights/safetensors.zig");
const vw = @import("vision_weights.zig");
const vision_sam = @import("vision_sam.zig");
const vision_clip = @import("vision_clip.zig");

pub const Error = vw.Error;
pub const Diagnostic = vw.Diagnostic;
pub const Size = vw.Size;
pub const Work = vw.Work;
pub const lm_hidden = vision_clip.lm_hidden;

/// One image's views: its global view, bf16 [3, 1024, 1024], and its crops, bf16 [w * h, 3, 640, 640] in
/// `images_spatial_crop`'s order (none where the image is not cropped).
pub const Image = struct {
    global: []const u16,
    crops: []const u16 = &.{},
    /// `images_spatial_crop`: crops a row, rows
    width: usize = 1,
    height: usize = 1,

    pub fn cropCount(self: Image) usize {
        return if (self.crops.len == 0) 0 else self.width * self.height;
    }

    /// The image's rows in `lm_features`.
    pub fn rows(self: Image) usize {
        const g = Size.global.out();
        const c = Size.crop.out();
        const local = if (self.cropCount() == 0) 0 else self.height * c * (self.width * c + 1);
        return local + g * (g + 1) + 1;
    }
};

/// The int32 tables of a size for its most views a run: the windows' maps and the attention's tiles.
const SizeLayout = struct {
    to_window: cuda.Buffer,
    tiles_window: cuda.Buffer,
    tiles_global: cuda.Buffer,
    tiles_clip: cuda.Buffer,

    fn build(gpa: Allocator, made: *vw.Made, size: Size, views: usize) Error!SizeLayout {
        var self: SizeLayout = undefined;
        const to = try vision_sam.windowMaps(gpa, size, views);
        defer gpa.free(to);
        self.to_window = try upload(made, to, &.{to.len});
        const win = try vision_sam.tiles(gpa, views * size.windows() * size.windows(), 196, 14);
        defer gpa.free(win);
        self.tiles_window = try upload(made, win, &.{ win.len / 4, 4 });
        const global = try vision_sam.tiles(gpa, views, size.tokens(), size.grid());
        defer gpa.free(global);
        self.tiles_global = try upload(made, global, &.{ global.len / 4, 4 });
        const clip = try vision_sam.tiles(gpa, views, 1 + size.outTokens(), 0);
        defer gpa.free(clip);
        self.tiles_clip = try upload(made, clip, &.{ clip.len / 4, 4 });
        return self;
    }

    fn upload(made: *vw.Made, values: []const i32, shape: []const usize) Error!cuda.Buffer {
        const buffer = try made.buffer(.i32, shape);
        try buffer.upload(std.mem.sliceAsBytes(values));
        return buffer;
    }

    /// The prefixes for `views` views of `size`.
    fn of(self: SizeLayout, size: Size, views: usize) vision_sam.Layout {
        const per_window = (196 + 127) / 128;
        const per_global = (size.tokens() + 127) / 128;
        return .{
            .to_window = self.to_window.span(0, views * size.windowRows()),
            .tiles_window = self.tiles_window.span(0, views * size.windows() * size.windows() * per_window),
            .tiles_global = self.tiles_global.span(0, views * per_global),
        };
    }

    fn clipTiles(self: SizeLayout, size: Size, views: usize) cuda.View {
        return self.tiles_clip.span(0, views * ((1 + size.outTokens() + 127) / 128));
    }
};

pub const Tower = struct {
    made: vw.Made,
    sam: vision_sam.Sam,
    clip: vision_clip.Clip,
    newline: cuda.Buffer,
    separator: cuda.Buffer,
    /// zeros [1024]: the bias of the convs that have none
    zeros: cuda.Buffer,
    work: Work,
    global_layout: SizeLayout,
    crop_layout: SizeLayout,
    crops_per_run: usize,
    globals_per_run: usize,
    /// int32 [2, n]: the place tables of an `encode` (grown as asked)
    places: ?cuda.Buffer = null,

    pub const Capacity = struct {
        crops_per_run: usize = 16,
        globals_per_run: usize = 4,
    };

    /// Uploads the tower from `checkpoint`'s `model.sam_model.`, `model.vision_model.`, `model.projector.`,
    /// `model.image_newline` and `model.view_seperator` tensors as stored (BF16), the tables of a crop resampled
    /// on the host; refuses with `diagnostic` set where a tensor is missing or not BF16 of its shape.
    pub fn load(gpa: Allocator, arena: Allocator, checkpoint: *const safetensors.Checkpoint, capacity: Capacity, diagnostic: *Diagnostic) Error!Tower {
        var ranges: std.ArrayList([]const u8) = .empty;
        defer ranges.deinit(gpa);
        for (checkpoint.tensors.keys(), checkpoint.tensors.values()) |name, tensor| {
            if (std.mem.startsWith(u8, name, "model.sam_model.") or std.mem.startsWith(u8, name, "model.vision_model.") or
                std.mem.startsWith(u8, name, "model.projector.") or std.mem.eql(u8, name, "model.image_newline") or
                std.mem.eql(u8, name, "model.view_seperator")) try ranges.append(gpa, tensor.data);
        }
        var held = try checkpoint.hold(gpa, ranges.items);
        defer held.release();

        const source: vw.Source = .{ .checkpoint = checkpoint, .arena = arena, .diagnostic = diagnostic };
        var self: Tower = undefined;
        self.made = .{ .gpa = gpa };
        errdefer self.made.deinit();
        self.places = null;
        self.crops_per_run = @max(capacity.crops_per_run, 1);
        self.globals_per_run = @max(capacity.globals_per_run, 1);
        self.zeros = try self.made.buffer(.bf16, &.{vision_sam.features});
        try self.zeros.zero();
        self.sam = try .load(gpa, source, &self.made, self.zeros.view);
        self.clip = try .load(gpa, source, &self.made);
        self.newline = try source.upload(&self.made, "model.image_newline", &.{lm_hidden});
        self.separator = try source.upload(&self.made, "model.view_seperator", &.{lm_hidden});

        const runs = [_]struct { size: Size, views: usize }{
            .{ .size = Size.global, .views = self.globals_per_run },
            .{ .size = Size.crop, .views = self.crops_per_run },
        };
        var tokens: usize = 0;
        var rel: usize = 0;
        var clip_rows: usize = 0;
        for (runs) |r| {
            const n = r.views * r.size.tokens();
            tokens = @max(tokens, n);
            rel = @max(rel, @max(r.views * r.size.windowRows() * 28, n * 2 * r.size.grid()));
            clip_rows = @max(clip_rows, r.views * (1 + r.size.outTokens()));
        }
        const wide = @max(@max(tokens * 3 * vision_sam.hidden, tokens * vision_sam.inner), clip_rows * vision_clip.inner);
        self.work = .{
            .pixels = try self.made.buffer(.bf16, &.{ tokens, vision_sam.hidden }),
            .h = try self.made.buffer(.bf16, &.{ tokens, vision_sam.hidden }),
            .y = try self.made.buffer(.bf16, &.{ tokens, vision_sam.hidden }),
            .branch = try self.made.buffer(.bf16, &.{ tokens, vision_sam.hidden }),
            .wide = try self.made.buffer(.bf16, &.{wide}),
            .rel = try self.made.buffer(.bf16, &.{ vision_sam.heads, rel }),
            .sam = try self.made.buffer(.bf16, &.{ tokens / 16, vision_sam.features }),
            .clip = try self.made.buffer(.f32, &.{ clip_rows, vision_clip.hidden }),
        };
        self.global_layout = try .build(gpa, &self.made, Size.global, self.globals_per_run);
        self.crop_layout = try .build(gpa, &self.made, Size.crop, self.crops_per_run);
        return self;
    }

    pub fn deinit(self: *Tower) void {
        if (self.places) |*p| p.deinit();
        self.made.deinit();
        self.* = undefined;
    }

    fn layoutOf(self: *const Tower, size: Size) SizeLayout {
        return if (size.side == Size.global.side) self.global_layout else self.crop_layout;
    }

    pub fn perRun(self: *const Tower, size: Size) usize {
        return if (size.side == Size.global.side) self.globals_per_run else self.crops_per_run;
    }

    /// SAM over `views` views of `size` whose pixels are in `work.pixels`: `work.sam` (`samView`).
    pub fn runSam(self: *const Tower, size: Size, views: usize) Error!void {
        std.debug.assert(views <= self.perRun(size));
        return self.sam.forward(self.work, self.layoutOf(size).of(size, views), size, views);
    }

    /// CLIP over the run's `work.sam`: `work.clip`.
    pub fn runClip(self: *const Tower, size: Size, views: usize) Error!void {
        std.debug.assert(views <= self.perRun(size));
        return self.clip.forward(self.work, self.layoutOf(size).clipTiles(size, views), size, views);
    }

    /// The concat and the projector over the run: bf16 [views * out tokens, 1280] in `work.branch`.
    pub fn runProject(self: *const Tower, size: Size, views: usize) Error!cuda.View {
        return self.clip.project(self.work, size, views);
    }

    /// Copies `views` views' pixels (bf16 [views, 3, S, S]) in for a run.
    pub fn putPixels(self: *const Tower, size: Size, pixels: []const u16) Error!void {
        std.debug.assert(pixels.len % (3 * size.side * size.side) == 0);
        try self.work.pixels.uploadAt(0, std.mem.sliceAsBytes(pixels));
    }

    /// SAM's features of the run, bf16 [views * out tokens, 1024], channels last.
    pub fn samView(self: *const Tower, size: Size, views: usize) cuda.View {
        return vw.shaped(self.work.sam, .bf16, &.{ views * size.outTokens(), vision_sam.features });
    }

    /// CLIP's output of the run, fp32 [views * (1 + out tokens), 1024].
    pub fn clipView(self: *const Tower, size: Size, views: usize) cuda.View {
        return vw.shaped(self.work.clip, .f32, &.{ views * (1 + size.outTokens()), vision_clip.hidden });
    }

    /// The images' rows, `lm_features` of each in turn, into `out` bf16 [rows, 1280] (rows the sum of
    /// `Image.rows`).
    pub fn encode(self: *Tower, gpa: Allocator, images: []const Image, out: cuda.View) Error!void {
        var total: usize = 0;
        for (images) |image| total += image.rows();
        try cuda.bound(out.ndim == 2 and @as(usize, @intCast(out.size[0])) == total and out.size[1] == lm_hidden, "vision: out is not [{d}, {d}]", .{ total, lm_hidden });
        const crop_values = 3 * Size.crop.side * Size.crop.side;
        const global_values = 3 * Size.global.side * Size.global.side;
        for (images) |image| {
            try cuda.bound(image.global.len == global_values, "vision: a global view of {d} values, not {d}", .{ image.global.len, global_values });
            try cuda.bound(image.crops.len == image.cropCount() * crop_values, "vision: {d} crop values for a {d} x {d} grid", .{ image.crops.len, image.width, image.height });
        }

        // where each view's rows go: from (a run's projector row, -1 newline, -2 separator) to (a row of out),
        // the crops' content first, by run, then the globals', then the newlines and separators
        var from: std.ArrayList(i32) = .empty;
        defer from.deinit(gpa);
        var to: std.ArrayList(i32) = .empty;
        defer to.deinit(gpa);
        var marks: std.ArrayList(i32) = .empty;
        defer marks.deinit(gpa);
        var marks_to: std.ArrayList(i32) = .empty;
        defer marks_to.deinit(gpa);
        try from.ensureTotalCapacity(gpa, total);
        try to.ensureTotalCapacity(gpa, total);
        {
            const c = Size.crop.out();
            const g = Size.global.out();
            var base: usize = 0;
            var crop_view: usize = 0;
            // the crops' content, in the order the runs take the crops (images in turn)
            for (images) |image| {
                const width = image.width * c + 1;
                for (0..image.cropCount()) |i| {
                    const cy = i / image.width;
                    const cx = i % image.width;
                    const run_row = (crop_view % self.crops_per_run) * c * c;
                    for (0..c) |y| for (0..c) |x| {
                        from.appendAssumeCapacity(@intCast(run_row + y * c + x));
                        to.appendAssumeCapacity(@intCast(base + (cy * c + y) * width + cx * c + x));
                    };
                    crop_view += 1;
                }
                if (image.cropCount() > 0) for (0..image.height * c) |r| {
                    try marks.append(gpa, -1);
                    try marks_to.append(gpa, @intCast(base + r * width + width - 1));
                };
                base += image.rows();
            }
            base = 0;
            for (images, 0..) |image, n| {
                const local = image.rows() - g * (g + 1) - 1;
                const run_row = (n % self.globals_per_run) * g * g;
                for (0..g) |y| {
                    for (0..g) |x| {
                        from.appendAssumeCapacity(@intCast(run_row + y * g + x));
                        to.appendAssumeCapacity(@intCast(base + local + y * (g + 1) + x));
                    }
                    try marks.append(gpa, -1);
                    try marks_to.append(gpa, @intCast(base + local + y * (g + 1) + g));
                }
                try marks.append(gpa, -2);
                try marks_to.append(gpa, @intCast(base + image.rows() - 1));
                base += image.rows();
            }
            try from.appendSlice(gpa, marks.items);
            try to.appendSlice(gpa, marks_to.items);
            std.debug.assert(from.items.len == total);
        }
        if (self.places == null or @as(usize, @intCast(self.places.?.view.size[1])) < total) {
            if (self.places) |*p| p.deinit();
            self.places = null;
            self.places = try cuda.Buffer.init(.i32, &.{ 2, total });
        }
        const places = self.places.?;
        const width: usize = @intCast(places.view.size[1]);
        try places.uploadAt(0, std.mem.sliceAsBytes(from.items));
        try places.uploadAt(width * 4, std.mem.sliceAsBytes(to.items));
        const from_all: cuda.View = .init(places.view.ptr.?, .i32, &.{total});
        const to_all: cuda.View = .init(@ptrCast(@as([*]u8, @ptrCast(places.view.ptr.?)) + width * 4), .i32, &.{total});

        var placed: usize = 0;
        // the crops, `crops_per_run` a run, images in turn
        {
            var crop_count: usize = 0;
            for (images) |image| crop_count += image.cropCount();
            var done: usize = 0;
            var image_at: usize = 0;
            var crop_at: usize = 0;
            while (done < crop_count) {
                const views = @min(self.crops_per_run, crop_count - done);
                var filled: usize = 0;
                while (filled < views) {
                    const image = images[image_at];
                    const take = @min(views - filled, image.cropCount() - crop_at);
                    if (take > 0) try self.work.pixels.uploadAt(filled * crop_values * 2, std.mem.sliceAsBytes(image.crops[crop_at * crop_values ..][0 .. take * crop_values]));
                    filled += take;
                    crop_at += take;
                    if (crop_at == image.cropCount()) {
                        image_at += 1;
                        crop_at = 0;
                    }
                }
                placed += try self.runPlaced(Size.crop, views, from_all, to_all, placed, out);
                done += views;
            }
        }
        // the global views
        {
            var done: usize = 0;
            while (done < images.len) {
                const views = @min(self.globals_per_run, images.len - done);
                for (0..views) |v| try self.work.pixels.uploadAt(v * global_values * 2, std.mem.sliceAsBytes(images[done + v].global));
                placed += try self.runPlaced(Size.global, views, from_all, to_all, placed, out);
                done += views;
            }
        }
        // the newlines and separators
        const marked = total - placed;
        try vw.place(null, from_all.span(placed, marked), to_all.span(placed, marked), self.newline.view, self.separator.view, out);
    }

    /// One run of `views` views whose pixels are in, its projector rows placed by the tables' entries from
    /// `first`: the rows it placed.
    fn runPlaced(self: *Tower, size: Size, views: usize, from: cuda.View, to: cuda.View, first: usize, out: cuda.View) Error!usize {
        try self.runSam(size, views);
        try self.runClip(size, views);
        const rows = try self.runProject(size, views);
        const n = views * size.outTokens();
        try vw.place(rows, from.span(first, n), to.span(first, n), self.newline.view, self.separator.view, out);
        return n;
    }
};
