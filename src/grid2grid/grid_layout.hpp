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

    int num_cols() const noexcept { return grid.num_cols(); }
    int num_rows() const noexcept { return grid.num_rows(); }

    assigned_grid2D grid;
    local_blocks<T> blocks;
};

} // namespace grid2grid
