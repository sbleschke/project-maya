// include/strata/core/qwen_exl3.hpp - an EXL3 pack of Qwen3.8-Flash-Next (tools/qwen/pack.py) on the `maya` engine's Qwen
// path: its matrices on the device as the weight table's native rows (exl3::kTypeEXL3), its routed experts on the CPU.
//
// load_experts puts the most-routed experts that fit in VRAM; layer_hits computes those on the device, inside the token
// graph, and adds them to the shared expert's output, so the combine that follows needs no change.  Any other expert is
// read from the model's files for every token and computed by the CPU lane's kernels, one worker a routed expert.
#pragma once

#include "strata/core/exl3_table.hpp"
#include "strata/kernels/exl3.hpp"

#include <cstdint>
#include <deque>
#include <fstream>
#include <set>
#include <string>
#include <utility>
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
    /// The experts of `profile` (ranked pairs; then every other expert, layer by layer) into `budget` bytes of VRAM.
    bool load_experts(const std::vector<std::pair<int32_t, int32_t>>& profile, uint64_t budget, int k, std::string& err);
    int64_t resident() const { return resident_; }
    int64_t experts() const { return (int64_t) table_.layers.size() * n_expert_; }
    uint64_t expert_vram_bytes() const { return expert_vram_; }
    /// The prompt path: starts[e] = the first row of expert e's rows, given how many each has (cnt) - the experts in VRAM
    /// first, in the order of their blobs, then the others by id.
    void row_starts(int layer, const int32_t* cnt, int32_t* starts) const;
    /// Every routed expert of `layer` over its rows: H16 [rows][n_ff] FP16 scratch, y [rows][n_embd] (each row r: x's row
    /// row_tok[r], `row_tok` and `x` device buffers).  The experts not in VRAM come through a small staging buffer.
    bool prompt_experts(int layer, const int32_t* cnt, const int32_t* starts, const float* x, const int32_t* row_tok,
                        uint16_t* h16, float* y, int64_t& n_resident, int64_t& n_streamed, void* stream, std::string& err);
    /// (a token graph's layer) shared += sum of weights[i] * expert ids[i] over x, for the experts in VRAM.  ids, weights,
    /// x and shared are device buffers; k experts.
    bool layer_hits(int layer, const int32_t* ids, const float* weights, const float* x, float* shared, void* stream,
                    std::string& err) const;
    /// `out` [k][n_embd]: routed experts `ids` of `layer` over x (n_embd floats), unweighted, as the pool's parts - zeros
    /// for an expert in VRAM (layer_hits has counted it).
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
    int n_expert_ = 0, k_ = 0;
    int64_t resident_ = 0;
    uint64_t expert_vram_ = 0;
    std::vector<unsigned long long> host_table_;      // [layer * n_expert + expert]: the blob in VRAM, 0: not there
    unsigned long long* d_table_ = nullptr;
    unsigned long long* d_plan_ = nullptr;            // [k]
    std::vector<strata::kernels::exl3::ExpertLayout> lay_;   // [layer]
    std::vector<const uint8_t*> arena_;               // [layer]: its experts in VRAM, ascending id, a blob apart
    std::vector<std::vector<int>> slots_;             // [layer]: those experts' ids
    int* d_bounds_ = nullptr;                         // [n_expert + 1]
    std::vector<int> host_bounds_;
    uint8_t* d_stage_ = nullptr;                      // kStage blobs
    std::vector<uint8_t> host_stage_;
    std::vector<std::vector<std::ifstream>> stage_files_;   // [reader][file]
    uint64_t max_blob_ = 0;
};

}  // namespace strata::core
