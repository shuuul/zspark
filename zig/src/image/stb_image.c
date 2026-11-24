// stb_image's JPEG decoder, compiled once for zig/src/image/jpeg.zig (spec 019 WS-07), and its BMP, TGA, PSD and
// PNM (P5, P6) decoders for zig/src/image/stb.zig (WS-04), from memory; PNG, GIF and WebP have their own
// decoders, and the size is checked by the Zig side (`zs_image_info`) before any picture is decoded.
#include <limits.h>
#include <stddef.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_ONLY_PSD
#define STBI_ONLY_PNM
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_FAILURE_STRINGS
#define STBI_MAX_DIMENSIONS 8192
#if defined(__aarch64__)
#define STBI_NEON
#endif
#include "stb_image.h"

// The picture as RGB rows into `out` (width * height * 3 bytes); 0, or 1 where stb_image refused the bytes or read
// another size than the markers gave.
int zs_image_jpeg(const unsigned char *data, size_t len, unsigned char *out, int width, int height) {
    if (len > INT_MAX) return 1;
    int w = 0, h = 0, n = 0;
    unsigned char *px = stbi_load_from_memory(data, (int)len, &w, &h, &n, 3);
    if (!px) return 1;
    int same = w == width && h == height;
    if (same) memcpy(out, px, (size_t)w * (size_t)h * 3);
    stbi_image_free(px);
    return same ? 0 : 1;
}

// The size and channels stb_image reads from the header of a BMP, TGA, PSD or PNM picture: 1, or 0 where none of
// its decoders takes the bytes.
int zs_image_info(const unsigned char *data, size_t len, int *width, int *height, int *channels) {
    if (len > INT_MAX) return 0;
    return stbi_info_from_memory(data, (int)len, width, height, channels);
}

// Only TGA may use the signatureless fallback; another decoder recognizing a header is not enough.
int zs_image_is_tga(const unsigned char *data, size_t len) {
    if (len > INT_MAX) return 0;
    return stbi_info2_from_memory(data, (int)len).type == STBI_type_tga;
}

// What stb_image reads through: the bytes, and whether it asked for one past their end, which it would read as
// zeros where Pillow refuses a truncated file ("image file is truncated").
typedef struct {
    const unsigned char *data;
    size_t len, at;
    int past;
} zs_reader;

static int zs_read(void *user, char *out, int size) {
    zs_reader *r = (zs_reader *)user;
    size_t left = r->len - r->at;
    if (left == 0) {
        r->past = 1;
        return 0;
    }
    size_t n = (size_t)size < left ? (size_t)size : left;
    memcpy(out, r->data + r->at, n);
    r->at += n;
    return (int)n;
}

static void zs_skip(void *user, int n) {
    zs_reader *r = (zs_reader *)user;
    size_t left = r->len - r->at;
    if (n > 0 && (size_t)n > left) {
        r->past = 1;
        r->at = r->len;
    } else if (n > 0) {
        r->at += (size_t)n;
    }
}

static int zs_eof(void *user) {
    zs_reader *r = (zs_reader *)user;
    return r->at >= r->len;
}

// The picture as `channels` (3 or 4) bytes a pixel into `out`; 0, or 1 where stb_image refused the bytes, read
// another size than its header gave, or ran past their end.
int zs_image_stb(const unsigned char *data, size_t len, unsigned char *out, int width, int height, int channels) {
    if (len > INT_MAX) return 1;
    zs_reader r = {data, len, 0, 0};
    stbi_io_callbacks io = {zs_read, zs_skip, zs_eof};
    int w = 0, h = 0, n = 0;
    unsigned char *px = stbi_load_from_callbacks(&io, &r, &w, &h, &n, channels);
    if (!px) return 1;
    int same = w == width && h == height && !r.past;
    if (same) memcpy(out, px, (size_t)w * (size_t)h * (size_t)channels);
    stbi_image_free(px);
    return same ? 0 : 1;
}
