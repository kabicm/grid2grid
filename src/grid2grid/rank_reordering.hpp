#pragma once
#include <grid2grid/comm_volume.hpp>
#include <unordered_set>
#include <vector>

namespace grid2grid {
std::vector<int> optimal_reordering(comm_volume& comm_volume, int n_ranks) {
    std::unordered_set<int> visited;

    // identity permutation
    std::vector<int> permutation;
    permutation.reserve(n_ranks);
    for (size_t i = 0; i < n_ranks; ++i) {
        permutation.push_back(i);
    }

    std::vector<weighted_edge_t> sorted_edges;
    sorted_edges.reserve(comm_volume.size());
    for (const auto& el : comm_volume.volume) {
        auto& e = el.first;
        int w = el.second;
        int src = e.src;
        int dest = e.dest;
        sorted_edges.push_back(weighted_edge_t(src, dest, w));
    }

    // sort the edges by weights (decreasing order)
    std::sort(sorted_edges.rbegin(), sorted_edges.rend());

    for (const auto& edge : sorted_edges) {
        // edge: src->dest with weight w
        if (visited.find(edge.src) != visited.end()) {
            continue;
        }
        if (visited.find(edge.dest) != visited.end()) {
            continue;
        }

        // map src -> dest
        // take this edge to perfect matching
        permutation[edge.src] = edge.dest;

        // no adjecent edge to these vertices
        // can be taken in the future
        // to preserve the perfect matching
        visited.insert(edge.src);
        visited.insert(edge.dest);
    }

    return permutation;
}
}


