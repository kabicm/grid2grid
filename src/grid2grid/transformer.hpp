#pragma once
#include <vector>
#include <cassert>
#include <mpi.h>
#include <grid2grid/transform.hpp>

namespace grid2grid {
template <typename T>
struct transformer {
    std::vector<layout_ref<T>> from;
    std::vector<layout_ref<T>> to;
    MPI_Comm comm;
    int P;
    int rank;

    // constructor
    transformer() = default;

    // constructor
    transformer(MPI_Comm comm) : comm(comm) {
        MPI_Comm_size(comm, &P);
        MPI_Comm_rank(comm, &rank);
    }

    void schedule(grid_layout<T>& from_layout, grid_layout<T>& to_layout) {
        from.push_back(from_layout);
        to.push_back(to_layout);
    }

    void transform() {
        grid2grid::transform<T>(from, to, comm);
        // for (unsigned i = 0u; i < from.size(); ++i) {
        //     grid2grid::transform<T>(from[i], to[i], comm);
        // }
        clear();
    }

    void clear() {
        from.clear();
        to.clear();
    }
};
}
