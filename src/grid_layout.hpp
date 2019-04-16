#pragma once
#include "grid2D.hpp"
#include "block.hpp"
#include "mpi_type_wrapper.hpp"

namespace grid2grid {
template <typename T>
class grid_layout {
public:
    grid_layout() = default;

    grid_layout(assigned_grid2D&& g, local_blocks<T>&& b):
        grid(std::forward<assigned_grid2D>(g)), blocks(std::forward<local_blocks<T>>(b)) {}

    int num_ranks() const {
        return grid.num_ranks();
    }

    assigned_grid2D grid;
    local_blocks<T> blocks;
};
}
