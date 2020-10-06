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

    // scaling scalars for source and destination
    std::vector<T> alpha;
    std::vector<T> beta;

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

    void schedule(grid_layout<T>& from_layout, grid_layout<T>& to_layout,
                  T alpha, T beta) {
        this->alpha.push_back(alpha);
        this->beta.push_back(beta);
        schedule(from_layout, to_layout);
    }

    void transform() {
        assert(alpha.size() == beta.size());
        if (alpha.size() > 0 && beta.size() > 0) {
            grid2grid::transform<T>(from, to, &alpha[0], &beta[0], comm);
        } else {
            grid2grid::transform<T>(from, to, comm);
        }
        clear();
    }

    void clear() {
        from.clear();
        to.clear();
        alpha.clear();
        beta.clear();
    }
};
}
