// src/core/qwen_exl3.cpp - see include/strata/core/qwen_exl3.hpp.
#include "strata/core/qwen_exl3.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <thread>

#include "glm_fast_state.hpp"

namespace strata::core {

namespace x3 = strata::kernels::exl3;
namespace xc = strata::kernels::cpu;

namespace {
constexpr size_t kRamAlign = 2u << 20;   // (a huge page; a blob's pieces, multiples of 256 bytes, keep their alignment)
// The CPU's experts: one pool for every GPU's object (a layer split runs its layers in turn).  It lives as long as the
// process; its workers spin 2 ms for the next batch, longer than a layer takes, because a futex wake a batch - four a
// layer - would cost more than a band.  Made by start_workers, at the start: physical_cores() keeps to the calling
// thread's CPUs, and the host loop's thread is pinned to one, where it would find no core for a worker.
glmfast::Workers& cpu_workers() {
    static glmfast::Workers* w = [] {
        const std::vector<int> cores = xc::physical_cores(true);
        return new glmfast::Workers((int) cores.size(), 2000, cores);
    }();
    return *w;
}
}  // namespace

namespace {
// `bytes` from a file into a new device allocation.
bool upload(const std::string& path, uint64_t off, uint64_t bytes, void*& out, std::vector<uint8_t>& buf, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    buf.resize(bytes);
    f.seekg((std::streamoff) off);
    f.read((char*) buf.data(), (std::streamsize) bytes);
    if (!f) {
        err = "EXL3 pack: cannot read " + std::to_string(bytes) + " bytes at " + std::to_string(off) + " of " + path;
        return false;
    }
    if (cudaMalloc(&out, bytes) != cudaSuccess || cudaMemcpy(out, buf.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "EXL3 pack: cannot upload " + std::to_string(bytes) + " bytes to the GPU";
        return false;
    }
    return true;
}
}  // namespace

QwenExl3::~QwenExl3() {
    for (void* p : device_) cudaFree(p);
    if (scratch_) cudaFree(scratch_);
    if (d_table_) cudaFree(d_table_);
    if (d_plan_) cudaFree(d_plan_);
    if (d_bounds_) cudaFree(d_bounds_);
    if (d_stage_) cudaFree(d_stage_);
    if (ram_) ::operator delete(ram_, std::align_val_t(kRamAlign));
}

bool QwenExl3::served_names(const std::string& pack_dir, std::set<std::string>& out, std::string& err) {
    Exl3Table t;
    if (!exl3_table(pack_dir, t, err)) {
        if (err.empty()) err = "EXL3 pack: no exl3.txt in " + pack_dir;
        return false;
    }
    for (const auto& kv : t.mats) out.insert(kv.first);
    out.insert("token_embd.weight");
    return true;
}

bool QwenExl3::load(const std::string& pack_dir, int n_expert, WeightTable& table, std::string& err) {
    if (!exl3_table(pack_dir, table_, err, n_expert)) {
        if (err.empty()) err = "EXL3 pack: no exl3.txt in " + pack_dir;
        return false;
    }
    dir_ = std::filesystem::path(pack_dir).parent_path().string();
    n_expert_ = n_expert;
    const auto path = [&](int file) { return (std::filesystem::path(dir_) / table_.files[(size_t) file]).string(); };
    std::vector<uint8_t> buf;
    int max_in = 0;
    for (const auto& kv : table_.mats) {
        const Exl3Table::Mat& m = kv.second;
        if (m.cb != 2) {
            err = "EXL3 pack: " + kv.first + " is not of the mul1 codebook";
            return false;
        }
        x3::Mat M;
        void* tr = nullptr;
        void* suh = nullptr;
        void* svh = nullptr;
        if (!upload(path(m.file), m.tr, m.tr_bytes(), tr, buf, err)) return false;
        device_.push_back(tr);
        if (!upload(path(m.file), m.suh, 2 * (uint64_t) m.k, suh, buf, err)) return false;
        device_.push_back(suh);
        if (!upload(path(m.file), m.svh, 2 * (uint64_t) m.n, svh, buf, err)) return false;
        device_.push_back(svh);
        M.trellis = (const uint32_t*) tr;
        M.suh = (const uint16_t*) suh;
        M.svh = (const uint16_t*) svh;
        M.k = m.k;
        M.n = m.n;
        M.k2 = m.k2;
        M.cb = m.cb;
        if (!x3::supported(M)) {
            err = "EXL3 pack: the kernels do not take " + kv.first;
            return false;
        }
        mats_.push_back(M);
        vram_ += m.vram();
        max_in = std::max(max_in, m.k);
    }
    if (cudaMalloc(&scratch_, strata::kernels::native_q8_1_bytes(max_in)) != cudaSuccess) {
        err = "EXL3 pack: cannot allocate the activation scratch";
        return false;
    }
    size_t i = 0;
    for (const auto& kv : table_.mats) {
        const x3::Mat& M = mats_[i++];
        auto found = table.table_.find(kv.first);
        if (found == table.table_.end()) {
            err = "EXL3 pack: " + kv.first + " is in exl3.txt but not in index.txt";
            return false;
        }
        WeightRef& ref = found->second;
        if (ref.ne0 != M.k || ref.ne1 != M.n) {
            err = "EXL3 pack: " + kv.first + "'s shape differs between exl3.txt and index.txt";
            return false;
        }
        ref.native_data = &M;
        ref.native_type = x3::kTypeEXL3;
        ref.native_q8_1 = scratch_;
        // (as iq_pack.py writes a row served natively: the layer's checks read a native row's form)
        ref.code_bits = 8;
        ref.group_elems = 32;
    }
    return true;
}

const void* QwenExl3::mat(const std::string& name) const {
    size_t i = 0;
    for (const auto& kv : table_.mats) {
        if (kv.first == name) return &mats_[i];
        ++i;
    }
    return nullptr;
}

namespace {
constexpr int kStage = 16;   // the prompt path's experts not in VRAM, a read and a launch at a time
// One blob: [gate | up | down], a part [trellis | suh | svh], read from the model's files (each piece a multiple of 256
// bytes, as the device kernels' layout wants).
bool read_blob(const Exl3Table::Layer& L, int id, std::vector<std::ifstream>& files, uint8_t* at) {
    for (int r = 0; r < 3; ++r) {
        const uint64_t bytes[3] = {(uint64_t) L.k[r] * L.n[r] * L.k2[r] / 16, 2 * (uint64_t) L.k[r], 2 * (uint64_t) L.n[r]};
        for (int j = 0; j < 3; ++j) {
            const size_t ix = (size_t) id * 9 + (size_t) r * 3 + (size_t) j;
            std::ifstream& f = files[(size_t) L.file[ix]];
            f.seekg((std::streamoff) L.off[ix]);
            f.read((char*) at, (std::streamsize) bytes[j]);
            if (!f) { f.clear(); return false; }
            at += bytes[j];
        }
    }
    return true;
}
}  // namespace

uint64_t QwenExl3::blob_bytes(int layer) const {
    const auto it = table_.layers.find(layer);
    return it == table_.layers.end() ? 0 : it->second.blob();
}

bool QwenExl3::load_experts(const std::vector<std::pair<int32_t, int32_t>>& profile, uint64_t budget, int k, std::string& err,
                            int lb, int le) {
    namespace x3k = strata::kernels::exl3;
    k_ = k;
    lb_ = lb;
    const int n_layers = table_.layers.empty() ? 0 : table_.layers.rbegin()->first + 1;
    le_ = le < 0 ? n_layers : le;
    host_table_.assign((size_t) n_layers * n_expert_, 0ull);
    lay_.assign((size_t) n_layers, x3k::ExpertLayout{});
    for (const auto& kv : table_.layers) {
        const Exl3Table::Layer& L = kv.second;
        x3k::ExpertLayout& lay = lay_[(size_t) kv.first];
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            lay.trellis[r] = at;
            at += (size_t) L.k[r] * L.n[r] * L.k2[r] / 16;
            lay.suh[r] = at;
            at += 2 * (size_t) L.k[r];
            lay.svh[r] = at;
            at += 2 * (size_t) L.n[r];
            lay.k2[r] = L.k2[r];
        }
        lay.cb = L.cb;
        if (L.cb != 2) {
            err = "EXL3 pack: layer " + std::to_string(kv.first) + "'s experts are not of the mul1 codebook";
            return false;
        }
    }
    // which ones: the profile's ranking, then the rest in order, until the budget is spent (a layer's blobs its own size)
    std::vector<std::vector<int>> chosen((size_t) n_layers);
    std::vector<char> taken((size_t) n_layers * n_expert_, 0);
    uint64_t used = 0;
    const auto take = [&](int l, int e) {
        const auto it = table_.layers.find(l);
        if (l < 0 || e < 0 || e >= n_expert_ || it == table_.layers.end() || taken[(size_t) l * n_expert_ + e]) return true;
        if (used + it->second.blob() > budget) return false;
        used += it->second.blob();
        taken[(size_t) l * n_expert_ + e] = 1;
        chosen[(size_t) l].push_back(e);
        return true;
    };
    arena_.assign((size_t) n_layers, nullptr);
    bool room = true;
    for (const auto& p : profile)
        if (!(room = take(p.first, p.second))) break;
    for (int l = lb; room && l < (le < 0 ? n_layers : le); ++l)
        for (int e = 0; room && e < n_expert_; ++e) room = take(l, e);
    // read a layer at a time on several threads, then one upload (the model's files are read once)
    const int n_threads = (int) std::clamp(std::thread::hardware_concurrency() / 2u, 2u, 16u);
    std::vector<std::vector<std::ifstream>> readers((size_t) n_threads);
    for (auto& r : readers)
        for (const std::string& name : table_.files)
            r.emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
    std::vector<uint8_t> host;
    for (int l = 0; l < n_layers; ++l) {
        std::vector<int>& es = chosen[(size_t) l];
        std::sort(es.begin(), es.end());   // (ascending id: a prompt's rows run in this order)
        if (es.empty()) continue;
        const Exl3Table::Layer& L = table_.layers.at(l);
        const uint64_t blob = L.blob(), bytes = blob * es.size();
        host.resize(bytes);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; ++t)
            workers.emplace_back([&, t] {
                for (size_t i; (i = next.fetch_add(1)) < es.size();)
                    if (!read_blob(L, es[i], readers[(size_t) t], host.data() + i * blob)) failed = true;
            });
        for (std::thread& w : workers) w.join();
        if (failed) {
            err = "EXL3 pack: a read of layer " + std::to_string(l) + "'s experts failed";
            return false;
        }
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (d) cudaFree(d);
            err = "EXL3 pack: cannot upload layer " + std::to_string(l) + "'s experts to the GPU";
            return false;
        }
        device_.push_back(d);
        arena_[(size_t) l] = (const uint8_t*) d;
        for (size_t i = 0; i < es.size(); ++i)
            host_table_[(size_t) l * n_expert_ + es[i]] = (unsigned long long) ((uint8_t*) d + i * blob);
        resident_ += (int64_t) es.size();
        expert_vram_ += bytes;
    }
    if (cudaMalloc((void**) &d_table_, host_table_.size() * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemcpy(d_table_, host_table_.data(), host_table_.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMalloc((void**) &d_plan_, (size_t) k * sizeof(unsigned long long)) != cudaSuccess) {
        err = "EXL3 pack: cannot allocate the experts' table on the GPU";
        return false;
    }
    slots_ = std::move(chosen);
    for (const auto& kv : table_.layers) max_blob_ = std::max(max_blob_, kv.second.blob());
    if (cudaMalloc((void**) &d_bounds_, (size_t) (n_expert_ + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_stage_, (size_t) kStage * max_blob_) != cudaSuccess) {
        err = "EXL3 pack: cannot allocate the prompt path's staging buffer on the GPU";
        return false;
    }
    return true;
}

int QwenExl3::start_workers() { return cpu_workers().size(); }

void QwenExl3::load_ram(const std::vector<std::pair<int32_t, int32_t>>& profile) {
    host_ram_.assign(host_table_.size(), nullptr);
    // what this PC's RAM holds, less a headroom; MAYA_EXPERT_RAM_GB caps all the GPUs' objects together
    static uint64_t taken = 0;
    uint64_t room = expert_ram_room(nullptr, nullptr, false);
    if (const char* v = std::getenv("MAYA_EXPERT_RAM_GB")) {
        const double gb = std::atof(v);
        if (gb >= 0.0) {
            const uint64_t cap = (uint64_t) (gb * 1073741824.0);
            room = std::min(room, cap > taken ? cap - taken : 0);
        }
    }
    // which ones: those not in VRAM, the profile's ranking, then the rest of this object's layers
    struct Pick { int l, e; uint64_t at; };
    std::vector<Pick> picks;
    std::vector<char> seen(host_table_.size(), 0);
    uint64_t want = 0;
    const auto take = [&](int l, int e) {
        const auto it = table_.layers.find(l);
        if (l < lb_ || l >= le_ || e < 0 || e >= n_expert_ || it == table_.layers.end()) return true;
        const size_t i = (size_t) l * n_expert_ + e;
        if (seen[i] || host_table_[i] != 0ull) return true;
        if (want + it->second.blob() > room) return false;
        seen[i] = 1;
        picks.push_back({l, e, want});
        want += it->second.blob();
        return true;
    };
    bool fits = true;
    for (const auto& p : profile)
        if (!(fits = take(p.first, p.second))) break;
    for (int l = lb_; fits && l < le_; ++l)
        for (int e = 0; fits && e < n_expert_; ++e) fits = take(l, e);
    if (picks.empty()) return;
    ram_ = (uint8_t*) ::operator new(want, std::align_val_t(kRamAlign), std::nothrow);
    if (ram_ == nullptr) {
        std::fprintf(stderr, "maya generate: no %.1f GiB of RAM for the experts: they are read from the model's files\n",
                     (double) want / 1073741824.0);
        return;
    }
#if defined(__linux__)
    madvise(ram_, want, MADV_HUGEPAGE);
#endif
    const int n_threads = (int) std::clamp(std::thread::hardware_concurrency() / 2u, 2u, 16u);
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t)
        workers.emplace_back([&] {
            std::vector<std::ifstream> files;
            for (const std::string& name : table_.files)
                files.emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
            for (size_t i; !failed && (i = next.fetch_add(1)) < picks.size();)
                if (!read_blob(table_.layers.at(picks[i].l), picks[i].e, files, ram_ + picks[i].at)) failed = true;
        });
    for (std::thread& w : workers) w.join();
    if (failed) {
        std::fprintf(stderr, "maya generate: a read of the experts into RAM failed: they are read from the model's files\n");
        ::operator delete(ram_, std::align_val_t(kRamAlign));
        ram_ = nullptr;
        return;
    }
    for (const Pick& p : picks) host_ram_[(size_t) p.l * n_expert_ + p.e] = ram_ + p.at;
    ram_resident_ = (int64_t) picks.size();
    ram_bytes_ = want;
    taken += want;
}

void QwenExl3::row_starts(int layer, const int32_t* cnt, int32_t* starts) const {
    int32_t at = 0;
    for (int e : slots_[(size_t) layer]) {
        starts[e] = at;
        at += cnt[e];
    }
    for (int e = 0; e < n_expert_; ++e)
        if (host_table_[(size_t) layer * n_expert_ + e] == 0ull) {
            starts[e] = at;
            at += cnt[e];
        }
}

bool QwenExl3::prompt_experts(int layer, const int32_t* cnt, const int32_t* starts, const float* x, const int32_t* row_tok,
                              uint16_t* h16, float* y, int64_t& n_resident, int64_t& n_streamed, void* stream,
                              std::string& err) {
    // A test hook: STRATA_X3_PROMPT_FAIL=<n> makes the n-th call here fail, once, counted over all the GPUs of a layer
    // split, so the engine's recovery can be checked (the request must end with an error and the engine must live).
    static std::atomic<int64_t> fail_in = [] {
        const char* v = std::getenv("STRATA_X3_PROMPT_FAIL");
        return std::atomic<int64_t>(v ? std::atoll(v) : 0);
    }();
    if (fail_in.load() > 0 && fail_in.fetch_sub(1) == 1) {
        std::fprintf(stderr, "maya exl3: layer %d's prompt experts cannot run (STRATA_X3_PROMPT_FAIL)\n", layer);
        err = "EXL3 pack: the prompt's experts of layer " + std::to_string(layer) + " ran out of memory";
        return false;
    }
    namespace x3k = strata::kernels::exl3;
    const cudaStream_t cs = (cudaStream_t) stream;
    const Exl3Table::Layer& L = table_.layers.at(layer);
    const int n_embd = L.k[0], n_ff = L.n[0];
    const auto launch = [&](const uint8_t* wbase, size_t stride, int n_exp, int row0, int max_rows) {
        x3k::SetArgs sa;
        sa.wbase = wbase;
        sa.stride = stride;
        sa.lay = lay_[(size_t) layer];
        sa.n_exp = n_exp;
        sa.bounds = d_bounds_;
        sa.row0 = row0;
        sa.max_rows = max_rows;
        sa.n_embd = n_embd;
        sa.n_ff = n_ff;
        x3k::set_gate_up(sa, x, n_embd, row_tok + row0, h16 + (size_t) row0 * n_ff, n_ff, cs);
        x3k::set_down(sa, h16 + (size_t) row0 * n_ff, n_ff, y + (size_t) row0 * n_embd, n_embd, cs);
    };
    // the experts in VRAM: one set
    const std::vector<int>& slots = slots_[(size_t) layer];
    host_bounds_.resize((size_t) n_expert_ + 1);
    int max_rows = 0, rows = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        host_bounds_[i] = rows;
        rows += cnt[slots[i]];
        max_rows = std::max(max_rows, cnt[slots[i]]);
        n_resident += cnt[slots[i]] > 0;
    }
    host_bounds_[slots.size()] = rows;
    if (rows > 0) {
        if (cudaMemcpy(d_bounds_, host_bounds_.data(), (slots.size() + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "EXL3 pack: cannot upload the prompt's expert bounds";
            return false;
        }
        launch(arena_[(size_t) layer], (size_t) L.blob(), (int) slots.size(), 0, max_rows);
    }
    // the others: read, uploaded and run a staging buffer at a time
    std::vector<int> miss;
    for (int e = 0; e < n_expert_; ++e)
        if (cnt[e] > 0 && host_table_[(size_t) layer * n_expert_ + e] == 0ull) miss.push_back(e);
    if (miss.empty()) return true;
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        err = "EXL3 pack: the prompt's experts in VRAM failed";
        return false;
    }
    const size_t blob = (size_t) L.blob();
    host_stage_.resize((size_t) kStage * max_blob_);
    if (stage_files_.empty()) {
        stage_files_.resize((size_t) kStage);
        for (auto& r : stage_files_)
            for (const std::string& name : table_.files)
                r.emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
    }
    for (size_t j0 = 0; j0 < miss.size(); j0 += kStage) {
        const int m = (int) std::min(miss.size() - j0, (size_t) kStage);
        std::atomic<bool> failed{false};
        cpu_workers().run(m, [&](int i) {
            uint8_t* to = host_stage_.data() + (size_t) i * blob;
            const uint8_t* ram = host_ram_.empty() ? nullptr : host_ram_[(size_t) layer * n_expert_ + miss[j0 + (size_t) i]];
            if (ram != nullptr) std::memcpy(to, ram, blob);
            else if (!read_blob(L, miss[j0 + (size_t) i], stage_files_[(size_t) i], to)) failed = true;
        });
        if (failed) {
            err = "EXL3 pack: a read of layer " + std::to_string(layer) + "'s experts failed";
            return false;
        }
        const int row0 = starts[miss[j0]];
        int at = row0, most = 0;
        for (int i = 0; i < m; ++i) {
            host_bounds_[(size_t) i] = at;
            at += cnt[miss[j0 + i]];
            most = std::max(most, cnt[miss[j0 + i]]);
        }
        host_bounds_[(size_t) m] = at;
        if (cudaMemcpy(d_stage_, host_stage_.data(), (size_t) m * blob, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_bounds_, host_bounds_.data(), ((size_t) m + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "EXL3 pack: cannot upload the prompt's experts";
            return false;
        }
        launch(d_stage_, blob, m, row0, most);
        if (cudaStreamSynchronize(cs) != cudaSuccess) {
            err = "EXL3 pack: the prompt's staged experts failed";
            return false;
        }
        n_streamed += m;
    }
    return true;
}

bool QwenExl3::layer_hits(int layer, const int32_t* ids, const float* weights, const float* x, float* shared, void* stream,
                          std::string& err) const {
    namespace x3k = strata::kernels::exl3;
    if (d_table_ == nullptr || layer < 0 || (size_t) layer >= lay_.size()) {
        err = "EXL3 pack: no experts' table for layer " + std::to_string(layer);
        return false;
    }
    if (resident_ == 0) return true;
    const cudaStream_t cs = (cudaStream_t) stream;
    const Exl3Table::Layer& L = table_.layers.at(layer);
    x3k::plan_from_table(ids, d_table_ + (size_t) layer * n_expert_, k_, d_plan_, cs);
    // the kernels take at most 8 experts a launch: the rest in another, each adding to `shared` in place (an element is
    // read and written by one thread of the block that finishes it)
    for (int c = 0; c < k_; c += 8) {
        x3k::MoeArgs a;
        a.plan_ptr = d_plan_ + c;
        a.plan_w = weights + c;
        a.lay = lay_[(size_t) layer];
        a.k = std::min(k_ - c, 8);
        a.n_embd = L.k[0];
        a.n_ff = L.n[0];
        x3k::moe_gate_up(a, x, nullptr, cs);
        x3k::moe_down(a, nullptr, shared, shared, cs);
    }
    return true;
}

void QwenExl3::experts_run(int layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t n_embd, int64_t k, float* out) {
    const Exl3Table::Layer& L = table_.layers.at(layer);
    const strata::kernels::exl3::ExpertLayout& lay = lay_[(size_t) layer];
    const int n_ff = L.n[0];
    const int ff_bands = n_ff / 128, embd_bands = (int) n_embd / 128;
    struct Miss {
        const float* x;
        float* out;
        int id;
        const uint8_t* blob;
        xc::Exl3Part part[3];
        float* xh[3];
        float sum[3];
        float* g;
        float* u;
    };
    std::vector<Miss> miss;
    miss.reserve((size_t) (n_tok * k));
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t i = 0; i < k; ++i) {
            const int id = ids[t * k + i];
            float* o = out + (t * k + i) * n_embd;
            const size_t at = (size_t) layer * n_expert_ + id;
            if (!host_table_.empty() && host_table_[at] != 0ull) {
                std::memset(o, 0, (size_t) n_embd * sizeof(float));   // (layer_hits has it)
                ++routed_vram_;
                continue;
            }
            const uint8_t* ram = host_ram_.empty() ? nullptr : host_ram_[at];
            ++(ram != nullptr ? routed_ram_ : routed_disk_);
            miss.push_back(Miss{x + t * n_embd, o, id, ram, {}, {}, {}, nullptr, nullptr});
        }
    const int m = (int) miss.size();
    if (m == 0) return;
    // each one's scratch, on 64-byte boundaries; and, for one not in RAM, its bytes and the files to read them with
    const size_t floats = (size_t) (2 * n_embd + 3 * n_ff), each = floats * sizeof(float);
    work_.resize((size_t) m * each + 64);
    float* scratch = (float*) (((uintptr_t) work_.data() + 63) & ~(uintptr_t) 63);
    for (int i = 0; i < m; ++i) {
        Miss& w = miss[(size_t) i];
        float* f = scratch + (size_t) i * floats;
        w.xh[0] = f;
        w.xh[1] = f + n_embd;
        w.xh[2] = f + 2 * n_embd;
        w.g = w.xh[2] + n_ff;
        w.u = w.g + n_ff;
        if (w.blob == nullptr && files_.size() <= (size_t) i) {
            files_.resize((size_t) i + 1);
            blobs_.resize((size_t) i + 1);
        }
        if (w.blob == nullptr && files_[(size_t) i].empty())
            for (const std::string& name : table_.files)
                files_[(size_t) i].emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
    }
    const uint64_t blob_bytes = L.blob();
    glmfast::Workers& pool = cpu_workers();
    const auto band = [&](Miss& w, int r, int b, float* y) {
        xc::exl3_cpu_band(w.part[r], w.xh[r], w.sum[r], b * 128, y + (size_t) b * 128);
    };
    // the inputs (and an expert not in RAM read)
    pool.run(m, [&](int i) {
        Miss& w = miss[(size_t) i];
        if (w.blob == nullptr) {
            std::vector<uint8_t>& buf = blobs_[(size_t) i];
            buf.resize((size_t) blob_bytes + 64);
            uint8_t* at = (uint8_t*) (((uintptr_t) buf.data() + 63) & ~(uintptr_t) 63);
            if (!read_blob(L, w.id, files_[(size_t) i], at)) {
                std::memset(at, 0, (size_t) blob_bytes);
                std::fprintf(stderr, "maya generate: layer %d expert %d: a read of the model's files failed\n", layer, w.id);
            }
            w.blob = at;
        }
        for (int r = 0; r < 3; ++r) {
            w.part[r].trellis = (const uint32_t*) (w.blob + lay.trellis[r]);
            w.part[r].suh = (const uint16_t*) (w.blob + lay.suh[r]);
            w.part[r].svh = (const uint16_t*) (w.blob + lay.svh[r]);
            w.part[r].k = L.k[r];
            w.part[r].n = L.n[r];
            w.part[r].k2 = L.k2[r];
        }
        w.sum[0] = xc::exl3_cpu_prepare(w.part[0], w.x, w.xh[0]);
        w.sum[1] = xc::exl3_cpu_prepare(w.part[1], w.x, w.xh[1]);
    });
    // gate and up, a band a job
    pool.run(m * 2 * ff_bands, [&](int j) {
        Miss& w = miss[(size_t) (j / (2 * ff_bands))];
        const int r = (j / ff_bands) & 1;
        band(w, r, j % ff_bands, r == 0 ? w.g : w.u);
    });
    // silu(gate) * up, the down's input
    pool.run(m, [&](int i) {
        Miss& w = miss[(size_t) i];
        for (int c = 0; c < n_ff; ++c) w.g[c] = w.g[c] / (1.0f + std::exp(-w.g[c])) * w.u[c];
        w.sum[2] = xc::exl3_cpu_prepare(w.part[2], w.g, w.xh[2]);
    });
    pool.run(m * embd_bands, [&](int j) {
        Miss& w = miss[(size_t) (j / embd_bands)];
        band(w, 2, j % embd_bands, w.out);
    });
}

}  // namespace strata::core
