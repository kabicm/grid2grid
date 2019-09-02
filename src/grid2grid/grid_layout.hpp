#pragma once
#include <grid2grid/block.hpp>
#include <grid2grid/grid2D.hpp>
#include <grid2grid/mpi_type_wrapper.hpp>

namespace grid2grid {
template <typename T>
class grid_layout {
  public:
    grid_layout() = default;

    grid_layout(assigned_grid2D &&g, local_blocks<T> &&b)
        : grid(std::forward<assigned_grid2D>(g))
        , blocks(std::forward<local_blocks<T>>(b)) {}

    int num_ranks() const { return grid.num_ranks(); }

    void transpose_or_conjugate(char flag) {
        if (flag == 'T' || flag == 'C') {
            grid.transpose();
            blocks.transpose_or_conjugate(flag);
        }
    }

    void reorder_ranks(std::vector<int>& reordering) {
        grid.reorder_ranks(reordering);
        ranks_reordered_ = true;
    }

    int reordered_rank(int rank) const {
        return grid.reordered_rank(rank);
    }

    bool ranks_reordered() {
        return ranks_reordered_;
    }

    assigned_grid2D grid;
    local_blocks<T> blocks;
    bool ranks_reordered_ = false;
};

} // namespace grid2grid
