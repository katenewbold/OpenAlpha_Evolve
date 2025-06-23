#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include "graph.hh"
#include "treedec.hh"
#include "omp.h"

template <class Value>
struct HomomorphismCounting {
    Graph G, F;
    NiceTreeDecomposition NTD;

    HomomorphismCounting(Graph F_, Graph G_) : F(F_), G(G_) {
        for (auto &nbh: F.adj) sort(nbh.begin(), nbh.end());
        for (auto &nbh: G.adj) sort(nbh.begin(), nbh.end());
        NTD = niceTreeDecomposition(F);
    }
    struct VectorHash {
        size_t operator()(const std::vector<int> &x) const {
            constexpr size_t p = 1e9+7;
            size_t hash = 0;
            for(int i = 0; i < x.size(); ++i) {
                hash += p * hash + x[i];
            }
            return hash;
        }
    };
    using DPTable = std::unordered_map<std::vector<int>, Value, VectorHash>;
    Value run() {
        auto [I, X] = run(NTD.root);
        return I[std::vector<int>()];
    }
    std::pair<DPTable, std::vector<int>> run(int x) {
        DPTable I, J, K;
        std::vector<int> X;
        if (NTD.isLeaf(x)) {
            X = {NTD.vertex(x)};
            for (int a = 0; a < G.n; ++a) {
                std::vector<int> phi = {a};
                I[phi] = Value(1);
            }
        } else if (NTD.isIntroduce(x)) {
            std::tie(J, X) = run(NTD.child(x));
            int p = std::distance(X.begin(), std::lower_bound(X.begin(), X.end(), NTD.vertex(x)));
            std::vector<int> neighborPos;
            for (int idx = 0; idx < (int)X.size(); ++idx) {
                if (std::binary_search(F.adj[NTD.vertex(x)].begin(), F.adj[NTD.vertex(x)].end(), X[idx])) {
                    neighborPos.push_back(idx);
                }
            }
            for (auto &[phi, val] : J) {
                std::vector<int> candidateTargets(G.n);
                std::iota(candidateTargets.begin(), candidateTargets.end(), 0);
                for (int pos : neighborPos) {
                    std::vector<int> tmp;
                    for (int a : candidateTargets) {
                        if (std::binary_search(G.adj[phi[pos]].begin(), G.adj[phi[pos]].end(), a)) {
                            tmp.push_back(a);
                        }
                    }
                    candidateTargets.swap(tmp);
                }
                auto psi = phi;
                psi.insert(psi.begin() + p, 0);
                for (int a : candidateTargets) {
                    psi[p] = a;
                    I[psi] += val;
                }
            }
            X.insert(X.begin()+p, NTD.vertex(x));
        } else if (NTD.isForget(x)) {
            std::tie(J, X) = run(NTD.child(x));
            int p = std::distance(X.begin(), std::lower_bound(X.begin(), X.end(), NTD.vertex(x)));
            X.erase(X.begin() + p);
            for (auto [phi, val]: J) {
                auto psi = phi;
                psi.erase(psi.begin() + p);
                I[psi] += val;
            }
        } else if (NTD.isJoin(x)) {
            std::tie(J, X) = run(NTD.left(x));
            std::tie(K, X) = run(NTD.right(x));
            if (J.size() > K.size()) swap(J, K);
            for (auto& [phi, val]: J) {
                if (K.count(phi)) I[phi] = val * K[phi];
            }
        }
        return std::make_pair(I, X);
    }
};

template <class Value>
struct HomomorphismCountingTree {
    Graph F, G;

    HomomorphismCountingTree(Graph F_, Graph G_) : F(F_), G(G_) { }
    Value run() {
        auto hom_r = run(0, -1);
        return accumulate(hom_r.begin(), hom_r.end(), Value(0));
    }
    std::vector<Value> run(int x, int p) {
        std::vector<Value> hom_x(G.n, 1);
        for (int y: F.adj[x]) {
            if (y == p) continue;
            auto hom_y = run(y, x);
            for (int a = 0; a < G.n; ++a) {
                Value sum = Value(0);
                for (int b: G.adj[a]) sum += hom_y[b];
                hom_x[a] *= sum;
            }
        }
        return hom_x;
    }
};

template <class Value>
Value hom_(Graph F, Graph G) {
    if (isTree(F)) {
        return HomomorphismCountingTree<Value>(F, G).run();
    } else {
        return HomomorphismCounting<Value>(F, G).run();
    }
}

template <class Value>
Value hom(Graph F, Graph G) {
    Value value = Value(1);

    for (auto Fi: connectedComponents(F)) {
        Value value_i = Value(0);
        for (auto Gj: connectedComponents(G)) value_i += hom_<Value>(Fi, Gj);
        value *= value_i;
    }
    return value;
}
