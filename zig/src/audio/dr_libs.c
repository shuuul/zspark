// dr_libs' WAV, FLAC and MP3 decoders, compiled once for zig/src/audio/decode.zig (spec 138 WS-02), from memory and
// to f32, without stdio. One stream at a time through an opaque handle: open (its channels and rate), read frames
// into the caller's buffer until none come, close. The Zig side owns every sample buffer and checks the counts.
#include <stddef.h>
#include <stdlib.h>

#define DR_WAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#define DR_WAV_NO_WCHAR
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#define DR_FLAC_NO_WCHAR
#define DR_FLAC_NO_OGG
#define DR_MP3_IMPLEMENTATION
#define DR_MP3_NO_STDIO
#include "dr_wav.h"
#include "dr_flac.h"
#include "dr_mp3.h"

// The kinds `zs_audio_open` takes, as `decode.zig`'s `Kind` numbers them.
enum { zs_wav = 1, zs_flac = 2, zs_mp3 = 3 };

typedef struct {
    int kind;
    drwav wav;
    drflac *flac;
    drmp3 mp3;
} zs_audio;

// A stream over `data` (which must outlive it), its channels and rate; NULL where the decoder refused the bytes.
// `frames` is the header's count of frames a channel, 0 where the format gives none before decoding (MP3 without
// an Xing/Info frame); it bounds nothing, the reads do.
void *zs_audio_open(int kind, const unsigned char *data, size_t len, unsigned *channels, unsigned *rate,
                    unsigned long long *frames) {
    zs_audio *a = calloc(1, sizeof(zs_audio));
    if (!a) return NULL;
    a->kind = kind;
    switch (kind) {
    case zs_wav:
        if (!drwav_init_memory(&a->wav, data, len, NULL)) break;
        *channels = a->wav.channels, *rate = a->wav.sampleRate, *frames = a->wav.totalPCMFrameCount;
        return a;
    case zs_flac:
        a->flac = drflac_open_memory(data, len, NULL);
        if (!a->flac) break;
        *channels = a->flac->channels, *rate = a->flac->sampleRate, *frames = a->flac->totalPCMFrameCount;
        return a;
    case zs_mp3:
        if (!drmp3_init_memory(&a->mp3, data, len, NULL)) break;
        *channels = a->mp3.channels, *rate = a->mp3.sampleRate;
        *frames = a->mp3.totalPCMFrameCount == DRMP3_UINT64_MAX ? 0 : a->mp3.totalPCMFrameCount;
        return a;
    }
    free(a);
    return NULL;
}

// Up to `count` frames, interleaved, into `out` (count * channels floats); the frames read, 0 at the end.
unsigned long long zs_audio_read(void *handle, float *out, unsigned long long count) {
    zs_audio *a = handle;
    switch (a->kind) {
    case zs_wav: return drwav_read_pcm_frames_f32(&a->wav, count, out);
    case zs_flac: return drflac_read_pcm_frames_f32(a->flac, count, out);
    case zs_mp3: return drmp3_read_pcm_frames_f32(&a->mp3, count, out);
    }
    return 0;
}

void zs_audio_close(void *handle) {
    zs_audio *a = handle;
    if (!a) return;
    switch (a->kind) {
    case zs_wav: drwav_uninit(&a->wav); break;
    case zs_flac: drflac_close(a->flac); break;
    case zs_mp3: drmp3_uninit(&a->mp3); break;
    }
    free(a);
}
