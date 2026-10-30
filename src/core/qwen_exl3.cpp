// src/core/qwen_exl3.cpp - see include/strata/core/qwen_exl3.hpp.
#include "strata/core/qwen_exl3.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
    for (int64_t i = 0; i < k; ++i)
        workers.emplace_back([=, this] { expert((int) i, layer, ids[i], x, out + i * n_embd); });
    for (std::thread& t : workers) t.join();
}

}  // namespace strata::core
