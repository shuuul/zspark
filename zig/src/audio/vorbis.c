// stb_vorbis's Ogg Vorbis decoder, compiled once for zig/src/audio/decode.zig (spec 138 WS-02), from memory without
// stdio, behind the same three calls as dr_libs.c's: open (channels, rate), read interleaved f32 frames, close.
#include <limits.h>
#include <stddef.h>

#define STB_VORBIS_NO_STDIO
#define STB_VORBIS_NO_PUSHDATA_API
#include "stb_vorbis.c"

// A stream over `data` (which must outlive it), its channels and rate; NULL where stb_vorbis refused the bytes
// (an Ogg stream that is not Vorbis among them).
void *zs_vorbis_open(const unsigned char *data, size_t len, unsigned *channels, unsigned *rate) {
    if (len > INT_MAX) return NULL;
    int error = 0;
    stb_vorbis *v = stb_vorbis_open_memory(data, (int)len, &error, NULL);
    if (!v) return NULL;
    stb_vorbis_info info = stb_vorbis_get_info(v);
    *channels = (unsigned)info.channels, *rate = info.sample_rate;
    return v;
}

// Up to `count` frames, interleaved, into `out` (count * channels floats); the frames read, 0 at the end.
unsigned long long zs_vorbis_read(void *handle, float *out, unsigned long long count, unsigned channels) {
    stb_vorbis *v = handle;
    unsigned long long floats = count * channels;
    if (floats > INT_MAX) floats = INT_MAX / channels * channels;
    return (unsigned long long)stb_vorbis_get_samples_float_interleaved(v, (int)channels, out, (int)floats);
}

void zs_vorbis_close(void *handle) {
    if (handle) stb_vorbis_close(handle);
}
