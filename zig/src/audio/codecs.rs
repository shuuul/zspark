//! In-memory demux and decoding for the additional ASR formats. Zig owns the input and
//! the accumulated samples. Each call lends one decoded packet to Zig before reading the next.

use std::ffi::{c_int, c_void};
use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::OnceLock;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error;
use symphonia::core::formats::{TrackType, probe::Hint};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::units::{Duration, TimeBase};
use symphonia_opus::OpusDecoder;

const OK: c_int = 0;
const INVALID: c_int = 1;
const UNSUPPORTED: c_int = 2;
const STOPPED: c_int = 3;

type Samples = extern "C" fn(*mut c_void, *const f32, usize, u32, usize) -> bool;

fn codecs() -> &'static CodecRegistry {
    static CODECS: OnceLock<CodecRegistry> = OnceLock::new();
    CODECS.get_or_init(|| {
        let mut registry = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut registry);
        registry.register_audio_decoder::<OpusDecoder>();
        registry
    })
}

fn decode(bytes: &[u8], context: *mut c_void, samples: Samples) -> Result<c_int, Error> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(&Hint::new(), source, Default::default(), Default::default())
        .map_err(|_| Error::DecodeError("invalid audio container"))?;
    let Some(track) = format.default_track(TrackType::Audio) else {
        return Ok(INVALID);
    };
    let Some(params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
        return Ok(UNSUPPORTED);
    };
    let track_id = track.id;
    let time_base = track.time_base;
    // The Ogg demuxer exposes Opus pre-skip on the track, apart from packet trimming.
    let mut pre_skip = if bytes.starts_with(b"OggS") {
        track.delay.unwrap_or(0) as usize
    } else {
        0
    };
    let mut decoder = codecs().make_audio_decoder(params, &Default::default())?;
    let mut interleaved = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        let discarded = trim_frames(packet.trim_start, time_base, rate).min(decoded.frames());
        let skip = pre_skip.min(decoded.frames() - discarded);
        pre_skip -= skip;
        let first = discarded + skip;
        let last = decoded
            .frames()
            .saturating_sub(trim_frames(packet.trim_end, time_base, rate));
        if first >= last {
            continue;
        }
        interleaved.resize(decoded.samples_interleaved(), 0.0f32);
        decoded.copy_to_slice_interleaved(&mut interleaved);
        if !samples(
            context,
            interleaved[first * channels..].as_ptr(),
            last - first,
            rate,
            channels,
        ) {
            return Ok(STOPPED);
        }
    }
    Ok(OK)
}

fn trim_frames(trim: Duration, time_base: Option<TimeBase>, rate: u32) -> usize {
    let Some(base) = time_base else { return 0 };
    let numerator = u128::from(trim.get()) * u128::from(base.numer.get()) * u128::from(rate);
    let denominator = u128::from(base.denom.get());
    ((numerator + denominator / 2) / denominator)
        .try_into()
        .unwrap_or(usize::MAX)
}

/// The callback borrows its sample pointer for the call alone and returns false to stop decoding.
///
/// # Safety
/// `data` must point to `len` readable bytes, and `context` must stay valid for the callback
/// until decoding returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zs_audio_decode_codecs(
    data: *const u8,
    len: usize,
    context: *mut c_void,
    samples: Samples,
) -> c_int {
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    match catch_unwind(AssertUnwindSafe(|| decode(bytes, context, samples))) {
        Ok(Ok(status)) => status,
        Ok(Err(Error::Unsupported(_))) => UNSUPPORTED,
        _ => INVALID,
    }
}
