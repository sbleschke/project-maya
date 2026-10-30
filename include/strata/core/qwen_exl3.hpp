// include/strata/core/qwen_exl3.hpp - an EXL3 pack of Qwen3.8-Flash-Next (tools/qwen/pack.py) on the `maya` engine's Qwen
// path: its matrices on the device as the weight table's native rows (exl3::kTypeEXL3), its routed experts on the CPU.
//
// The experts are read from the model's files for every token and computed by the CPU lane's kernels, one worker a
// routed expert: the plain path that shows the model's numbers are right, before any expert cache or GPU expert.
#pragma once

#include "strata/core/exl3_table.hpp"
#include "strata/kernels/exl3.hpp"

#include <cstdint>
#include <deque>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

class WeightTable;

class QwenExl3 {
public:
    QwenExl3() = default;
    ~QwenExl3();
    QwenExl3(const QwenExl3&) = delete;
    QwenExl3& operator=(const QwenExl3&) = delete;

    /// The rows the weight table must not load: the EXL3 matrices and the embedding (NativeEmbed's).
    static bool served_names(const std::string& pack_dir, std::set<std::string>& out, std::string& err);
    /// Uploads every matrix and attaches it to its row of `table`; opens the experts' files.
    bool load(const std::string& pack_dir, int n_expert, WeightTable& table, std::string& err);
    /// A matrix by its row name (a const exl3::Mat*), or nullptr.
    const void* mat(const std::string& name) const;
    size_t mat_count() const { return mats_.size(); }
    uint64_t vram_bytes() const { return vram_; }
    /// `out` [k][n_embd]: routed experts `ids` of `layer` over x (n_embd floats), unweighted, as the pool's parts.
    void experts_run(int layer, const float* x, const int32_t* ids, int64_t n_embd, int64_t k, float* out);

private:
    void expert(int worker, int layer, int id, const float* x, float* out);

    Exl3Table table_;
    std::string dir_;
    std::deque<strata::kernels::exl3::Mat> mats_;
    std::vector<void*> device_;
    void* scratch_ = nullptr;
    uint64_t vram_ = 0;
    std::vector<std::vector<std::ifstream>> files_;   // [worker][file]
    std::vector<std::vector<uint8_t>> blobs_;         // [worker]
};

}  // namespace strata::core
