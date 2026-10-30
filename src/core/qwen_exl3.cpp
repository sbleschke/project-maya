// src/core/qwen_exl3.cpp - see include/strata/core/qwen_exl3.hpp.
#include "strata/core/qwen_exl3.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

namespace strata::core {

namespace x3 = strata::kernels::exl3;
namespace xc = strata::kernels::cpu;

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

bool QwenExl3::load_experts(const std::vector<std::pair<int32_t, int32_t>>& profile, uint64_t budget, int k, std::string& err) {
    namespace x3k = strata::kernels::exl3;
    k_ = k;
    const int n_layers = table_.layers.empty() ? 0 : table_.layers.rbegin()->first + 1;
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
    bool room = true;
    for (const auto& p : profile)
        if (!(room = take(p.first, p.second))) break;
    for (int l = 0; room && l < n_layers; ++l)
        for (int e = 0; room && e < n_expert_; ++e) room = take(l, e);
    // read a layer at a time on several threads, then one upload (the model's files are read once)
    const int n_threads = (int) std::clamp(std::thread::hardware_concurrency() / 2u, 2u, 16u);
    std::vector<std::vector<std::ifstream>> readers((size_t) n_threads);
    for (auto& r : readers)
        for (const std::string& name : table_.files)
            r.emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
    std::vector<uint8_t> host;
    for (int l = 0; l < n_layers; ++l) {
        const std::vector<int>& es = chosen[(size_t) l];
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

void QwenExl3::expert(int worker, int layer, int id, const float* x, float* out) {
    const Exl3Table::Layer& L = table_.layers.at(layer);
    auto& files = files_[(size_t) worker];
    auto& blob = blobs_[(size_t) worker];
    const auto align64 = [](uint8_t* p) { return (uint8_t*) (((uintptr_t) p + 63) & ~(uintptr_t) 63); };
    // the expert's three parts, each piece on a 64-byte boundary (the files keep them on none)
    uint64_t need = 0;
    for (int r = 0; r < 3; ++r) need += L.part(r) + 3 * 64;
    const int n_ff = L.n[0], n_embd = L.k[0];
    const uint64_t floats = 4 * (uint64_t) n_embd + 2 * (uint64_t) n_ff + 128;   // xh of each, gate, up
    blob.resize(need + floats * sizeof(float) + 64);
    uint8_t* at = align64(blob.data());
    xc::Exl3Part part[3];
    for (int r = 0; r < 3; ++r) {
        const uint64_t bytes[3] = {(uint64_t) L.k[r] * L.n[r] * L.k2[r] / 16, 2 * (uint64_t) L.k[r], 2 * (uint64_t) L.n[r]};
        uint8_t* piece[3];
        for (int j = 0; j < 3; ++j) {
            const size_t ix = (size_t) id * 9 + (size_t) r * 3 + (size_t) j;
            std::ifstream& f = files[(size_t) L.file[ix]];
            f.seekg((std::streamoff) L.off[ix]);
            f.read((char*) at, (std::streamsize) bytes[j]);
            if (!f) {
                f.clear();
                std::memset(at, 0, bytes[j]);
                std::fprintf(stderr, "maya generate: layer %d expert %d: a read of the model's files failed\n", layer, id);
            }
            piece[j] = at;
            at = align64(at + bytes[j]);
        }
        part[r].trellis = (const uint32_t*) piece[0];
        part[r].suh = (const uint16_t*) piece[1];
        part[r].svh = (const uint16_t*) piece[2];
        part[r].k = L.k[r];
        part[r].n = L.n[r];
        part[r].k2 = L.k2[r];
    }
    float* xh = (float*) at;
    float* g = xh + n_embd;
    float* u = g + n_ff;
    const auto run = [&](const xc::Exl3Part& m, const float* in, float* y) {
        const float sum = xc::exl3_cpu_prepare(m, in, xh);
        for (int c = 0; c < m.n; c += 128) xc::exl3_cpu_band(m, xh, sum, c, y + c);
    };
    run(part[0], x, g);
    run(part[1], x, u);
    for (int i = 0; i < n_ff; ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];   // silu(gate) * up
    run(part[2], g, out);
}

void QwenExl3::experts_run(int layer, const float* x, const int32_t* ids, int64_t n_embd, int64_t k, float* out) {
    if (files_.size() < (size_t) k) {
        const size_t had = files_.size();
        files_.resize((size_t) k);
        blobs_.resize((size_t) k);
        for (size_t w = had; w < (size_t) k; ++w)
            for (const std::string& name : table_.files)
                files_[w].emplace_back((std::filesystem::path(dir_) / name).string(), std::ios::binary);
    }
    std::vector<std::thread> workers;
    workers.reserve((size_t) k);
    for (int64_t i = 0; i < k; ++i) {
        if (!host_table_.empty() && host_table_[(size_t) layer * n_expert_ + ids[i]] != 0ull)
            std::memset(out + i * n_embd, 0, (size_t) n_embd * sizeof(float));   // (layer_hits has it)
        else
            workers.emplace_back([=, this] { expert((int) i, layer, ids[i], x, out + i * n_embd); });
    }
    for (std::thread& t : workers) t.join();
}

}  // namespace strata::core
