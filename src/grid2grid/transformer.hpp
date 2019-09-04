#pragma once
#include <vector>
#include <cassert>
#include <mpi.h>

#include <grid2grid/transform.hpp>

namespace grid2grid {
template <typename T>
struct transformer {
    std::vector<grid_layout<T>&> tasks;
    MPI_Comm comm;
    int P;
    int rank;
    // total communication volume for transformation of layouts
    comm_volume comm_vol;

    std::vector<int> rank_permutation;

    // constructor
    schedule() = default;

    // constructor
    schedule(MPI_Comm comm) : comm(comm) {
        MPI_Comm_size(comm, &P);
        MPI_Comm_rank(comm, &rank);
    }

    void schedule(grid_layout<T>& from_layout, grid_layout<T>& to_layout) {
        tasks.push_back(from_layout);
        tasks.push_back(to_layout);
    }

    void minimize_comm_volume(assigned_grid2D& from_grid, assigned_grid2D& to_grid) {
        // total communication volume for transformation of layouts
        comm_vol += communication_volume(from_grid, to_grid);
    }

    std::vector<int>& optimal_reordering() {
        if (rank_permutation.size() == 0) {
            // compute the optimal rank reordering that minimizes the communication volume
            rank_permutation = grid2grid::optimal_reordering(comm_vol, P);
        }
        return rank_permutation;
    }

    void transform() {
        assert(rank_permutation.size());

        assert(tasks.size() % 2 == 0);
        for (int i = 0; i < tasks.size()/2; ++i) {
            tasks[2*i+1].reorder_ranks(rank_permutation);
            transform(tasks[2*i], tasks[2*i+1]);
        }
        clear();
    }

    void clear() {
        tasks.clear();
        rank_permutation.clear();
    }
};
}
