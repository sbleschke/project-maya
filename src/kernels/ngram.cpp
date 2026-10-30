// src/kernels/ngram.cpp - P2.S4: the PLE n-gram hash and the IQ4_NL table read.
//
// See include/strata/kernels/ngram.hpp for the semantics, the rival readings, and the note on MADV_RANDOM.
#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <stdexcept>

#if defined(_WIN32)
// `PrefetchVirtualMemory` (memoryapi.h, Windows 8+) is the whole point of the change in `gather` below.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace strata::kernels {

namespace {
/// The A/B arm.  Host-token-path only, so it needs no atomics; see the note on `ple_prefetch_enable`.
bool g_ple_prefetch = true;
}  // namespace

void ple_prefetch_enable(bool on) { g_ple_prefetch = on; }
bool ple_prefetch_enabled() { return g_ple_prefetch; }

PleConsts ple_artifact_consts() {
    // docs/gguf-dump-shard1.txt, verbatim.  Written once here rather than derived, because `head_offsets` is
    // ALSO in the metadata as its own array - and deriving one from the other would hide a mismatch between
    // them instead of surfacing it.  The parity test checks the two agree.
    PleConsts c{};
    c.mult[0] = 23703573157769ull;
    c.mult[1] = 20109073645365ull;
    c.mult[2] = 8052911324071ull;
    const uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t offset[PLE_N_HEADS] = {
        0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    for (int i = 0; i < PLE_N_HEADS; ++i) {
        c.vocab[i] = vocab[i];
        c.offset[i] = offset[i];
    }
    return c;
}

uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    // The first term is an ASSIGNMENT and the rest are XORed into it, which is how the source writes it
    // (`uint64_t mixed = ctx[0]*m[0]; for j=1.. mixed ^= ctx[j]*m[j];`).  Every product wraps mod 2^64,
    // which is what the `(uint64_t)` casts in the source make explicit.
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is OLDEST FIRST, so predecessor `s` positions back is entry (n_prev - s): s=1 reads the
            // NEWEST.  Reading index (s-1) instead walks the window backwards, which still produces indices
            // in range and so cannot be caught by a range check - only by an oracle.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is
            // itself EOS.  Storing first and then cutting would leave position s holding the real token while
            // position s+1 became EOS - one token of history too much.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

namespace {
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

int iq4nl_code(int code) { return kIq4Nl[code & 15]; }

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j+16, not 2j and 2j+1.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

namespace {
/// All 65536 mul1 values, each rounded to fp16 (exllamav3's mul1_codebook).  The product of the 11-bit sum and the
/// 11-bit inverse is exact in F32, so an FMA and a separate add give the same bits.
const std::vector<float>& mul1_values() {
    static const std::vector<float> table = [] {
        std::vector<float> t(65536);
        const float k_inv = f32_from_f16(0x1eee), k_bias = f32_from_f16(0xc931);
        for (uint32_t s = 0; s < 65536; ++s) {
            const uint32_t x = s * 0x83DCD12Du;
            const uint32_t sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
            const float v = (1024.0f + (float) sum) * k_inv + k_bias;
            t[s] = f32_from_f16(f16_from_f32(v));
        }
        return t;
    }();
    return table;
}
}  // namespace

void ngram_trellis_dequant_row(const uint8_t* row, int K, const float* bias160, float* out160) {
    const std::vector<float>& cb = mul1_values();
    const int n_words = PLE_HEAD_DIM * K / 16;
    uint16_t w[1 + PLE_HEAD_DIM * PLE_TRELLIS_MAX_K / 16];
    std::memcpy(w, row, (size_t) (1 + n_words) * 2);
    const float scale = f32_from_f16(w[0]);
    const uint16_t* stream = w + 1;
    const uint32_t mask = (1u << K) - 1u;
    // the symbols, the last ones again in front: an element's state reaches back (15 + K) / K symbols, around the ring
    const int back = (15 + K) / K - 1;
    uint8_t sym_store[PLE_HEAD_DIM + 16];
    uint8_t* sym = sym_store + back;
    for (int i = 0; i < PLE_HEAD_DIM; ++i) {
        const int bit = i * K, at = bit >> 4;
        const uint32_t window = (uint32_t) stream[at] | ((uint32_t) stream[(at + 1) % n_words] << 16);
        sym[i] = (uint8_t) ((window >> (bit & 15)) & mask);
    }
    for (int j = 1; j <= back; ++j) sym[-j] = sym[PLE_HEAD_DIM - j];
    for (int i = 0; i < PLE_HEAD_DIM; ++i) {
        uint32_t state = 0;
        for (int j = 0; j <= back; ++j) state |= (uint32_t) sym[i - j] << (j * K);
        out160[i] = f32_from_f16(f16_from_f32(cb[state & 0xFFFFu] * scale + bias160[i]));
    }
}

// ---------------------------------------------------------------------------------------------------
struct PleTable::Impl {
    GgufFile* file = nullptr;
    const uint8_t* data = nullptr;
    uint64_t n_rows = 0;
    mutable uint64_t bytes_read = 0;
    // Direct mode (plan v0.3 P2): the mapping above is released after the header parse and every row comes
    // from an unbuffered SSD read into `raw`.
    PleIo mode = PleIo::Mmap;
    strata::ngram::PleReader reader;
    strata::ngram::PleReader::Ticket ticket;
    bool pending = false;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_MAX_ROW_BYTES] = {};
    // an EXL3 table (open_trellis): K != 0, rows of 2 + 20 K bytes, each decoded with its head's bias
    int K = 0;
    uint32_t row_bytes = PLE_ROW_BYTES;
    uint64_t head_offsets[PLE_N_HEADS] = {};
    std::vector<float> head_bias;

    void decode(uint32_t row, const uint8_t* bytes, float* out160) const {
        if (K == 0) {
            iq4nl_dequant_row(bytes, out160);
        } else if (row >= n_rows) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));   // (the IQ4_NL path's zeros: scale 0, no bias)
        } else {
            const int head = (int) (std::upper_bound(head_offsets, head_offsets + PLE_N_HEADS, (uint64_t) row) -
                                    head_offsets) - 1;
            ngram_trellis_dequant_row(bytes, K, head_bias.data() + (size_t) head * PLE_HEAD_DIM, out160);
        }
    }
};

PleTable::PleTable() : impl_(new Impl) {}
PleTable::~PleTable() { close(); delete impl_; }

bool PleTable::open(const std::string& gguf_path, std::string& err) {
    return open(gguf_path, err, PleIoOptions{});
}

bool PleTable::open_trellis(const std::string& txt_path, std::string& err, const PleIoOptions& io) {
    close();
    if (io.mode != PleIo::Direct) {
        err = "an EXL3 n-gram table is read Direct only (no --ple-io mmap)";
        return false;
    }
    std::ifstream txt(txt_path);
    if (!txt) {
        err = "cannot read " + txt_path;
        return false;
    }
    std::map<std::string, std::string> kv;
    for (std::string line; std::getline(txt, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const size_t sp = line.find(' ');
        if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
    }
    const auto number = [&](const char* key, uint64_t& out) {
        const auto it = kv.find(key);
        if (it == kv.end() || it->second.empty() || it->second.find_first_not_of("0123456789") != std::string::npos) {
            err = txt_path + ": no number for " + key;
            return false;
        }
        out = std::stoull(it->second);
        return true;
    };
    uint64_t K = 0, row_dim = 0, row_bytes = 0, n_rows = 0, heads = 0;
    uint64_t table_off = 0, bias_off = 0, offsets_off = 0, vocab_off = 0, mult_off = 0;
    if (!number("K", K) || !number("row_dim", row_dim) || !number("row_bytes", row_bytes) || !number("rows", n_rows) ||
        !number("heads", heads) || !number("table_off", table_off) || !number("head_bias_off", bias_off) ||
        !number("head_offsets_off", offsets_off) || !number("head_vocab_sizes_off", vocab_off) ||
        !number("layer_multipliers_off", mult_off))
        return false;
    if (kv["format"] != "exl3_ngram_trellis") {
        err = txt_path + ": format '" + kv["format"] + "', not exl3_ngram_trellis";
        return false;
    }
    if (K < 1 || K > (uint64_t) PLE_TRELLIS_MAX_K || row_dim != (uint64_t) PLE_HEAD_DIM ||
        row_bytes != 2 + row_dim * K / 8 || heads != (uint64_t) PLE_N_HEADS) {
        err = txt_path + ": the row geometry is not a 160-wide table of K 1..8 with 16 heads";
        return false;
    }
    std::string file = kv["file"];
    const bool absolute = !file.empty() && (file[0] == '/' || file[0] == '\\' || (file.size() > 1 && file[1] == ':'));
    const size_t slash = txt_path.find_last_of("/\\");
    if (!absolute && slash != std::string::npos) file = txt_path.substr(0, slash + 1) + file;
    std::ifstream f(file, std::ios::binary);
    const auto read_at = [&](uint64_t off, void* out, size_t n) {
        f.seekg((std::streamoff) off);
        f.read((char*) out, (std::streamsize) n);
        if (!f) err = file + ": cannot read the table's parameters";
        return (bool) f;
    };
    uint16_t bias[PLE_N_HEADS * PLE_HEAD_DIM];
    int64_t offsets[PLE_N_HEADS], vocab[PLE_N_HEADS], mult[NGRAM_SIZE];
    if (!f || !read_at(bias_off, bias, sizeof bias) || !read_at(offsets_off, offsets, sizeof offsets) ||
        !read_at(vocab_off, vocab, sizeof vocab) || !read_at(mult_off, mult, sizeof mult)) {
        if (err.empty()) err = "cannot open " + file;
        return false;
    }
    f.close();
    // (<=: the rows are rounded up to a multiple of 128, and the last few belong to no hash)
    if (offsets[PLE_N_HEADS - 1] + vocab[PLE_N_HEADS - 1] > (int64_t) n_rows) {
        err = file + ": the heads reach past the table's rows";
        return false;
    }
    if (!io.any_hash) {
        const PleConsts c = ple_artifact_consts();
        bool same = n_rows == PLE_TABLE_ROWS;
        for (int i = 0; i < PLE_N_HEADS; ++i)
            same = same && (uint64_t) offsets[i] == c.offset[i] && (uint64_t) vocab[i] == c.vocab[i];
        for (int i = 0; i < NGRAM_SIZE; ++i) same = same && (uint64_t) mult[i] == c.mult[i];
        if (!same) {
            err = file + ": its hash constants are not Qwen3.8-Flash-Next's: every row read would be the wrong one";
            return false;
        }
    }
    if (!impl_->reader.open(file, table_off, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread,
                            (uint32_t) row_bytes)) {
        close();
        return false;
    }
    impl_->head_bias.resize((size_t) PLE_N_HEADS * PLE_HEAD_DIM);
    for (size_t i = 0; i < impl_->head_bias.size(); ++i) impl_->head_bias[i] = f32_from_f16(bias[i]);
    for (int i = 0; i < PLE_N_HEADS; ++i) impl_->head_offsets[i] = (uint64_t) offsets[i];
    impl_->n_rows = n_rows;
    impl_->K = (int) K;
    impl_->row_bytes = (uint32_t) row_bytes;
    impl_->mode = PleIo::Direct;
    return true;
}

bool PleTable::open(const std::string& gguf_path, std::string& err, const PleIoOptions& io) {
    if (gguf_path.size() > 4 && gguf_path.compare(gguf_path.size() - 4, 4, ".txt") == 0)
        return open_trellis(gguf_path, err, io);
    close();
    try {
        impl_->file = new GgufFile(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const TensorInfo* t = impl_->file->find("per_layer_token_embd.weight");
    if (t == nullptr) {
        err = "per_layer_token_embd.weight is not in " + gguf_path;
        close();
        return false;
    }
    // [160, 320001536]: ne0 = 160 is the FAST axis, so the ROW index is shape[1] and a row is contiguous.
    if (t->shape.size() != 2 || t->shape[0] != (uint64_t) PLE_HEAD_DIM) {
        err = "per_layer_token_embd.weight has an unexpected shape";
        close();
        return false;
    }
    if (std::strcmp(t->type_name(), "IQ4_NL") != 0) {
        err = std::string("per_layer_token_embd.weight is ") + t->type_name() + ", not IQ4_NL";
        close();
        return false;
    }
    impl_->n_rows = t->shape[1];
    impl_->data = impl_->file->tensor_data(*t);

    // THE CHECK THAT MAKES THE OFFSET FALSIFIABLE.  The manifest's `shard2_tensor.offset` is 0, but that is
    // the offset within the GGUF's DATA SECTION: the file's first 192 bytes are a header, and reading at 0
    // would decode the header plus 192 bytes of shifted rows - still plausible IQ4_NL, and wrong for every
    // row.  `GgufFile` parses the header, so `tensor_data` is already correct; this asserts the tensor
    // exactly fills the file from there, which is what makes the whole arrangement checkable rather than
    // assumed.  A wrong data offset would leave a different remainder.
    // A shard may hold other tensors too (Swift 1.5's shard 1 holds layers 0-12 and the table): the table must
    // then fit inside the file at its own offset; alone in its shard (the original's shard 2) it fills it exactly.
    const uint64_t need = impl_->n_rows * (uint64_t) PLE_ROW_BYTES;
    const uint64_t have = impl_->file->file_size() - impl_->file->data_start();
    const bool alone = impl_->file->tensors().size() == 1;
    if (alone ? need != have : t->offset + need > have) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "PLE table size mismatch: %llu rows x %d B = %llu at offset %llu, but the file holds %llu from "
                      "data_start %llu",
                      (unsigned long long) impl_->n_rows, PLE_ROW_BYTES, (unsigned long long) need,
                      (unsigned long long) t->offset, (unsigned long long) have,
                      (unsigned long long) impl_->file->data_start());
        err = buf;
        close();
        return false;
    }
    if (io.mode == PleIo::Direct) {
        // The parse above is the validated source of the offset; the mapping itself is not kept, so no page of
        // the table can enter this process's working set or the file cache through it.
        const uint64_t table_offset = impl_->file->data_start() + t->offset;
        const uint64_t n_rows = impl_->n_rows;
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        if (!impl_->reader.open(gguf_path, table_offset, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread)) {
            close();
            return false;
        }
        impl_->n_rows = n_rows;
    }
    impl_->mode = io.mode;
    return true;
}

void PleTable::close() {
    impl_->reader.close();
    impl_->pending = false;
    impl_->mode = PleIo::Mmap;
    delete impl_->file;
    impl_->file = nullptr;
    impl_->data = nullptr;
    impl_->n_rows = 0;
    impl_->K = 0;
    impl_->row_bytes = PLE_ROW_BYTES;
    impl_->head_bias.clear();
}

bool PleTable::is_open() const { return impl_->data != nullptr || impl_->reader.is_open(); }
PleIo PleTable::mode() const { return impl_->mode; }
uint64_t PleTable::rows() const { return impl_->n_rows; }
uint64_t PleTable::bytes_read() const { return impl_->bytes_read; }

void PleTable::read_row(uint32_t row, float* out160) const {
    if (impl_->mode == PleIo::Direct && impl_->reader.is_open()) {
        uint8_t raw[PLE_MAX_ROW_BYTES];
        std::string err;
        const auto t = impl_->reader.issue(&row, 1, raw);
        if (!impl_->reader.collect(t, err)) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        impl_->decode(row, raw, out160);
        impl_->bytes_read += impl_->row_bytes;
        return;
    }
    if (impl_->data == nullptr || row >= impl_->n_rows) {
        std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        return;
    }
    iq4nl_dequant_row(impl_->data + (size_t) row * PLE_ROW_BYTES, out160);
    impl_->bytes_read += PLE_ROW_BYTES;
}

bool PleTable::issue(const uint32_t* rows16) {
    std::memcpy(impl_->rows, rows16, sizeof impl_->rows);
    if (impl_->mode == PleIo::Direct) {
        if (impl_->pending) return false;              // one token in flight per table
        impl_->ticket = impl_->reader.issue(impl_->rows, PLE_N_HEADS, impl_->raw);
        impl_->pending = true;
        return true;
    }
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * PLE_ROW_BYTES);
            ranges[n].NumberOfBytes = PLE_ROW_BYTES;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    impl_->pending = true;
    return true;
}

bool PleTable::collect(float* out2560, std::string& err) {
    if (!impl_->pending) { err = "PleTable::collect without issue"; return false; }
    impl_->pending = false;
    if (impl_->mode == PleIo::Direct) {
        if (!impl_->reader.collect(impl_->ticket, err)) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h)
            impl_->decode(impl_->rows[h], impl_->raw + (size_t) h * impl_->row_bytes, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * impl_->row_bytes;
        return true;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
    return true;
}

bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err) {
    if (impl_->pending) { err = "PleTable::gather_batch while a token is in flight"; return false; }
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    if (impl_->mode == PleIo::Direct) {
        std::vector<uint8_t> raw(n * impl_->row_bytes);
        const auto ticket = impl_->reader.issue(rows, n, raw.data());
        if (!impl_->reader.collect(ticket, err)) return false;
        for (size_t i = 0; i < n; ++i) impl_->decode(rows[i], raw.data() + i * impl_->row_bytes, out + i * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) n * impl_->row_bytes;
        return true;
    }
    for (size_t i = 0; i < n; ++i) read_row(rows[i], out + i * PLE_HEAD_DIM);
    return true;
}

void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

std::string PleTable::io_report() const {
    if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats& s = impl_->reader.stats();
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                  "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
                  (unsigned long long) s.requests, s.requests ? 100.0 * (double) s.cache_hits / (double) s.requests : 0.0,
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
    return buf;
}

void PleTable::gather(const uint32_t* rows16, float* out2560) const {
    if (impl_->mode == PleIo::Direct) {
        // `gather` stays const for its existing callers; the reader's state is the table's I/O state.
        PleTable* self = const_cast<PleTable*>(this);
        std::string err;
        if (!self->issue(rows16) || !self->collect(out2560, err)) {
            std::fprintf(stderr, "PleTable::gather: %s\n", err.empty() ? "a token is already in flight" : err.c_str());
            std::memset(out2560, 0, (size_t) NG_N_EMBD * sizeof(float));
        }
        return;
    }
    // ================================ SIXTEEN SERIAL PAGE FAULTS, MEASURED ================================
    //
    // **THIS COST 2.10-2.61 ms PER TOKEN AND HAD NEVER BEEN IN THE PLAN'S BUDGET AT ALL.**  The round-309
    // `token host phases` line put it second behind the layer loop among avoidable terms, and the arithmetic
    // says why: the table is 320,001,536 rows of `PLE_ROW_BYTES` = 90 B in a 26.8 GB mapping, so the sixteen
    // rows a token needs are 1,440 B - **0.5 MB/s**.  That is not bandwidth, it is latency: sixteen reads into
    // sixteen different 4 KB pages scattered across 26.8 GB, taken ONE AT A TIME, and on this machine the PLE
    // shard is 26.8 GB against 63 GB of RAM that the 31.6 GB expert arena is also competing for, so they are
    // not in the OS cache.  Sixteen serial NVMe reads at ~150 us is 2.4 ms, which is the measurement.
    //
    // `PrefetchVirtualMemory` issues all sixteen in ONE call and lets them complete in parallel.  It is a hint
    // and cannot change the answer - a range it does not fetch is simply faulted in by the read that follows -
    // so the only risk is that it does nothing.
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            // Out-of-range rows are handled by `read_row` as zeros and have no address to prefetch.
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * PLE_ROW_BYTES);
            ranges[n].NumberOfBytes = PLE_ROW_BYTES;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    // HEAD-SLOWEST, which is what `ggml_get_rows` does and what the source's own comment says: head h's 160
    // values occupy [h*160, (h+1)*160).  A head-fastest layout would put element (d, h) at d*16 + h and needs
    // a real transpose - a reshape of the same flat buffer compares equal and would make the check vacuous.
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace strata::kernels
