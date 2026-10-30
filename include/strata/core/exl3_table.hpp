// include/strata/core/exl3_table.hpp - a pack's exl3.txt, as tools/exl3_pack.py writes it (the GLM path, and the Qwen
// path through tools/qwen/pack.py).
#pragma once

#include "strata/kernels/exl3.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

// tools/exl3_pack.py's exl3.txt (an EXL3 pack; absent for a GGUF pack): the safetensors files, the EXL3 matrices
// (index rows of kind x) and every routed expert's nine pieces.  false + empty err: no such file.  Each matrix's and
// part's bitrate is K bits a weight, written "3", or "3.5" for exllamav3 1.6's half-integer K (1.5 / 2.5 / 3.5, mul1);
// carried as k2 = 2 K (strata::kernels::exl3::Mat::k2).  An engine from before half-integer K refuses such a line.
struct Exl3Table {
    std::vector<std::string> files;
    struct Mat {
        int file = 0;
        uint64_t tr = 0, suh = 0, svh = 0;
        int k = 0, n = 0, k2 = 0, cb = 2;
        uint64_t tr_bytes() const { return (uint64_t) k * n * k2 / 16; }
        uint64_t vram() const {   // as load_pack uploads it: each piece 256-aligned
            const auto up = [](uint64_t b) { return (b + 255u) & ~(uint64_t) 255u; };
            return up(tr_bytes()) + up(2 * (uint64_t) k) + up(2 * (uint64_t) n);
        }
    };
    std::map<std::string, Mat> mats;
    struct Layer {
        int cb = -1, k2[3] = {0, 0, 0}, k[3] = {0, 0, 0}, n[3] = {0, 0, 0};
        std::vector<int> file;        // [9 e + 3 r + j]
        std::vector<uint64_t> off;
        uint64_t part(int r) const {
            return (uint64_t) k[r] * n[r] * k2[r] / 16 + 2 * (uint64_t) k[r] + 2 * (uint64_t) n[r];
        }
        uint64_t blob() const { return 2 * part(0) + part(2); }
    };
    std::map<int, Layer> layers;
};

// an exl3.txt bitrate field as k2 = 2 K: "1".."8", or a half "1.5".."7.5" (the kernels take 1.5 / 2.5 / 3.5) - read
// by hand, so no locale's decimal point gets in the way
inline bool exl3_k2_field(const std::string& tok, int& k2) {
    if (tok.empty() || tok[0] < '1' || tok[0] > '8') return false;
    const int K = tok[0] - '0';
    if (tok.size() == 1) {
        k2 = 2 * K;
        return true;
    }
    if (tok.size() == 3 && tok[1] == '.' && tok[2] == '5' && K < 8) {
        k2 = 2 * K + 1;
        return true;
    }
    return false;
}
inline std::string exl3_k_name(int k2) { return (k2 & 1) ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2); }

inline bool exl3_table(const std::string& pack_dir, Exl3Table& t, std::string& err, int n_expert = 0) {
    err.clear();
    std::ifstream f(pack_dir + "/exl3.txt");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string what;
        ss >> what;
        if (what == "file") {
            int i = -1;
            std::string nm;
            if (!(ss >> i >> nm) || i != (int) t.files.size()) {
                err = "exl3.txt: bad file line: " + line;
                return false;
            }
            t.files.push_back(nm);
        } else if (what == "mat") {
            std::string nm, kt;
            Exl3Table::Mat m;
            if (!(ss >> nm >> m.file >> m.tr >> m.suh >> m.svh >> m.k >> m.n >> kt >> m.cb) || m.file < 0 ||
                m.file >= (int) t.files.size() || !exl3_k2_field(kt, m.k2)) {
                err = "exl3.txt: bad mat line: " + line;
                return false;
            }
            if (!strata::kernels::exl3::k2_supported(m.k2, m.cb)) {
                err = "exl3.txt: " + nm + ": K " + kt + " with codebook " + std::to_string(m.cb) +
                      " - the kernels take K 1..8 with any codebook and 1.5 / 2.5 / 3.5 with mul1 (2)";
                return false;
            }
            t.mats[nm] = m;
        } else if (what == "layer") {
            int il = -1;
            Exl3Table::Layer L;
            uint64_t gu = 0, dn = 0;
            std::string kt[3];
            if (!(ss >> il >> L.cb >> kt[0] >> kt[1] >> kt[2] >> L.k[0] >> L.n[0] >> L.k[2] >> L.n[2] >> gu >> dn) ||
                !exl3_k2_field(kt[0], L.k2[0]) || !exl3_k2_field(kt[1], L.k2[1]) || !exl3_k2_field(kt[2], L.k2[2])) {
                err = "exl3.txt: bad layer line: " + line;
                return false;
            }
            if (L.k2[0] != L.k2[1]) {   // the blob's layout (an expert's up after its gate, down after both) needs them alike
                err = "exl3.txt: layer " + std::to_string(il) + ": gate K " + kt[0] + " and up K " + kt[1] + " differ";
                return false;
            }
            for (int r = 0; r < 3; ++r)
                if (!strata::kernels::exl3::k2_supported(L.k2[r], L.cb)) {
                    err = "exl3.txt: layer " + std::to_string(il) + " " + (r == 0 ? "gate" : r == 1 ? "up" : "down") +
                          ": K " + kt[r] + " with codebook " + std::to_string(L.cb) +
                          " - the kernels take K 1..8 with any codebook and 1.5 / 2.5 / 3.5 with mul1 (2)";
                    return false;
                }
            L.k[1] = L.k[0];
            L.n[1] = L.n[0];
            if (L.part(0) != gu || L.part(2) != dn) {
                err = "exl3.txt: layer " + std::to_string(il) + "'s part sizes disagree with its shapes";
                return false;
            }
            if (n_expert > 0) {
                L.file.assign((size_t) n_expert * 9, -1);
                L.off.assign((size_t) n_expert * 9, 0);
            }
            t.layers[il] = std::move(L);
        } else if (what == "e") {
            if (n_expert <= 0) continue;   // the sizing passes need the layers only
            int il = -1, e = -1;
            ss >> il >> e;
            auto it = t.layers.find(il);
            if (it == t.layers.end() || e < 0 || e >= n_expert) {
                err = "exl3.txt: expert line for an unknown layer / expert: " + line;
                return false;
            }
            for (int j = 0; j < 9; ++j) {
                int fi = -1;
                uint64_t off = 0;
                if (!(ss >> fi >> off) || fi < 0 || fi >= (int) t.files.size()) {
                    err = "exl3.txt: bad expert line: " + line;
                    return false;
                }
                it->second.file[(size_t) e * 9 + (size_t) j] = fi;
                it->second.off[(size_t) e * 9 + (size_t) j] = off;
            }
        }
    }
    for (const auto& kv : t.layers)
        for (size_t i = 0; i < kv.second.file.size(); ++i)
            if (kv.second.file[i] < 0) {
                err = "exl3.txt: layer " + std::to_string(kv.first) + " lacks expert " + std::to_string(i / 9);
                return false;
            }
    return true;
}

}  // namespace strata::core
