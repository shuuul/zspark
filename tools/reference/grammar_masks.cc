// A C ABI over the part of xgrammar `grammar_masks.zig` calls. Exceptions stop here: every function that can
// fail writes a message into err (NUL-terminated, at most err_len bytes) and returns NULL or a negative value.
// Built by `grammar_masks.sh` against an xgrammar checkout; nothing of zspark links it.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

struct zxg_tokenizer { xgrammar::TokenizerInfo info; };
struct zxg_compiler { xgrammar::GrammarCompiler compiler; };
struct zxg_grammar { xgrammar::CompiledGrammar compiled; };
struct zxg_matcher { xgrammar::GrammarMatcher matcher; };

namespace {
void say(char* err, size_t err_len, const char* what) {
  if (err == nullptr || err_len == 0) return;
  std::strncpy(err, what, err_len - 1);
  err[err_len - 1] = 0;
}
template <typename T, typename F>
T* guarded(char* err, size_t err_len, F make) {
  try {
    return make();
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
  } catch (...) {
    say(err, err_len, "unknown C++ exception");
  }
  return nullptr;
}
}  // namespace

extern "C" {

// Token id is bytes[starts[id]..starts[id + 1]], already decoded (VocabType RAW); an empty token is special to
// xgrammar (never allowed). vocab_size is the logits' width.
zxg_tokenizer* zxg_tokenizer_new(const uint8_t* bytes, const uint32_t* starts, int32_t count, int32_t vocab_size,
                                 const int32_t* stop_ids, int32_t stops, char* err, size_t err_len) {
  return guarded<zxg_tokenizer>(err, err_len, [&] {
    std::vector<std::string> vocab;
    vocab.reserve(count);
    for (int32_t i = 0; i < count; ++i)
      vocab.emplace_back(reinterpret_cast<const char*>(bytes) + starts[i], starts[i + 1] - starts[i]);
    std::vector<int32_t> stop(stop_ids, stop_ids + stops);
    return new zxg_tokenizer{xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, vocab_size, stop, false)};
  });
}

zxg_compiler* zxg_compiler_new(const zxg_tokenizer* t, int32_t max_threads, int64_t cache_bytes, char* err,
                               size_t err_len) {
  return guarded<zxg_compiler>(err, err_len, [&] {
    return new zxg_compiler{xgrammar::GrammarCompiler(t->info, max_threads, true, cache_bytes)};
  });
}

// As TensorFold compiles a schema: any whitespace, no indent or separators fixed, strict mode. max_whitespace
// <= 0 is unbounded.
zxg_grammar* zxg_compile_json_schema(zxg_compiler* c, const char* schema, size_t len, int32_t max_whitespace,
                                     char* err, size_t err_len) {
  return guarded<zxg_grammar>(err, err_len, [&] {
    std::optional<int> blanks = max_whitespace > 0 ? std::optional<int>(max_whitespace) : std::nullopt;
    return new zxg_grammar{
        c->compiler.CompileJSONSchema(std::string(schema, len), true, std::nullopt, std::nullopt, true, blanks)};
  });
}
void zxg_grammar_free(zxg_grammar* g) { delete g; }

zxg_matcher* zxg_matcher_new(const zxg_grammar* g, char* err, size_t err_len) {
  return guarded<zxg_matcher>(err, err_len, [&] { return new zxg_matcher{xgrammar::GrammarMatcher(g->compiled)}; });
}
void zxg_matcher_free(zxg_matcher* m) { delete m; }

// words: (vocab_size + 31) / 32 int32, token t is bit t % 32 of word t / 32. 1: filled, 0: all allowed, <0: error.
int32_t zxg_fill(zxg_matcher* m, int32_t* words, int32_t count, char* err, size_t err_len) {
  try {
    int64_t shape = count;
    DLTensor t;
    t.data = words;
    t.device = DLDevice{kDLCPU, 0};
    t.ndim = 1;
    t.dtype = xgrammar::GetBitmaskDLType();
    t.shape = &shape;
    t.strides = nullptr;
    t.byte_offset = 0;
    if (m->matcher.FillNextTokenBitmask(&t)) return 1;
    std::memset(words, 0xff, sizeof(int32_t) * static_cast<size_t>(count));
    return 0;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
  } catch (...) {
    say(err, err_len, "unknown C++ exception");
  }
  return -1;
}
int32_t zxg_accept(zxg_matcher* m, int32_t token, char* err, size_t err_len) {
  try {
    return m->matcher.AcceptToken(token) ? 1 : 0;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
  } catch (...) {
    say(err, err_len, "unknown C++ exception");
  }
  return -1;
}
int32_t zxg_rollback(zxg_matcher* m, int32_t tokens, char* err, size_t err_len) {
  try {
    m->matcher.Rollback(tokens);
    return 0;
  } catch (const std::exception& e) {
    say(err, err_len, e.what());
  } catch (...) {
    say(err, err_len, "unknown C++ exception");
  }
  return -1;
}
int32_t zxg_terminated(const zxg_matcher* m) { return m->matcher.IsTerminated() ? 1 : 0; }
}
