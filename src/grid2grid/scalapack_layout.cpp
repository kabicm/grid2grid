#include <grid2grid/scalapack_layout.hpp>

namespace grid2grid {
namespace scalapack {

std::ostream &operator<<(std::ostream &os, const int_pair &other) {
    os << "[" << other.row << ", " << other.col << "]";
    return os;
}
rank_grid_coord
rank_to_grid(int rank, rank_decomposition grid_dim, ordering grid_ord) {
    if (rank < 0 || rank >= grid_dim.n_total()) {
        throw std::runtime_error(
            "Error in rank_to_grid: rank does not belong to the grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.row;
        return {rank % ld, rank / ld};
    } else {
        int ld = grid_dim.col;
        return {rank / ld, rank % ld};
    }
}

rank_grid_coord rank_to_grid(int rank,
                             rank_decomposition grid_dim,
                             ordering grid_ord,
                             rank_grid_coord src) {
    if (rank < 0 || rank >= grid_dim.n_total()) {
        throw std::runtime_error(
            "Error in rank_to_grid: rank does not belong to the grid.");
    }

    rank_grid_coord r_coord = rank_to_grid(rank, grid_dim, grid_ord);
    r_coord = (r_coord + src) % grid_dim;
    return r_coord;
}

int rank_from_grid(rank_grid_coord grid_coord,
                   rank_decomposition grid_dim,
                   ordering grid_ord) {
    if (grid_coord.row < 0 || grid_coord.row >= grid_dim.row ||
        grid_coord.col < 0 || grid_coord.col >= grid_dim.col) {
        throw std::runtime_error(
            "Error in rank_from_grid: rank coordinates do not belong \
    to the rank grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.row;
        return grid_coord.col * ld + grid_coord.row;
    } else {
        int ld = grid_dim.col;
        return grid_coord.row * ld + grid_coord.col;
    }
}

std::tuple<int, int> local_coordinate(int glob_coord,
                                      int block_dimension,
                                      int p_block_dimension,
                                      int mat_dim) {
    int idx_block = glob_coord / block_dimension;
    int idx_in_block = glob_coord % block_dimension;
    int idx_block_proc = idx_block / p_block_dimension;
    int owner = idx_block % p_block_dimension;

    return {idx_block_proc * block_dimension + idx_in_block, owner};
}

// global->local coordinates
local_grid_coord local_coordinates(matrix_grid mat_grid,
                                   rank_decomposition rank_grid,
                                   elem_grid_coord global_coord) {
    int row, p_row;
    std::tie<int, int>(row, p_row) =
        local_coordinate(global_coord.row,
                         mat_grid.block_dimension.row,
                         rank_grid.row,
                         mat_grid.matrix_dimension.row);
    int col, p_col;
    std::tie<int, int>(col, p_col) =
        local_coordinate(global_coord.col,
                         mat_grid.block_dimension.col,
                         rank_grid.col,
                         mat_grid.matrix_dimension.row);
    return {{row, col}, {p_row, p_col}};
}

// local->global coordinates
elem_grid_coord global_coordinates(matrix_grid mat_grid,
                                   rank_decomposition rank_grid,
                                   local_grid_coord local_coord) {
    int li = local_coord.el_coord.row;
    int lj = local_coord.el_coord.col;

    int my_row_rank = local_coord.rank_coord.row;
    int my_col_rank = local_coord.rank_coord.col;

    int num_row_ranks = rank_grid.row;
    int num_col_ranks = rank_grid.col;

    int mb = mat_grid.block_dimension.row;
    int nb = mat_grid.block_dimension.col;

    int gi = ((li / mb) * num_row_ranks + my_row_rank) * mb + (li % mb);
    int gj = ((lj / nb) * num_col_ranks + my_col_rank) * nb + (lj % nb);

    if (gi < 0 || gi > mat_grid.matrix_dimension.row || gj < 0 ||
        gj >= mat_grid.matrix_dimension.col) {
        // std::cout << "coordinates (" << gi << ", " << gj << ") should belong
        // to ("
        //     << mat_grid.matrix_dimension.row << ", " <<
        //     mat_grid.matrix_dimension.col << ")" << std::endl;
        // throw std::runtime_error("ERROR in scalapack::global_coordinates,
        // values out of range.");
        return {-1, -1};
    }
    return {gi, gj};
}

local_blocks get_local_blocks(matrix_grid mat_grid,
                              rank_decomposition r_grid,
                              rank_grid_coord rank_coord) {
    auto b_dim = mat_grid.block_dimension;
    auto m_dim = mat_grid.matrix_dimension;

    int n_blocks_row = (int)std::ceil(1.0 * m_dim.row / b_dim.row);
    int n_blocks_col = (int)std::ceil(1.0 * m_dim.col / b_dim.col);

    int n_owning_blocks_row =
        n_blocks_row / r_grid.row +
        (rank_coord.row < n_blocks_row % r_grid.row ? 1 : 0);
    int n_owning_blocks_col =
        n_blocks_col / r_grid.col +
        (rank_coord.col < n_blocks_col % r_grid.col ? 1 : 0);

    return {n_owning_blocks_row, n_owning_blocks_col, b_dim, rank_coord};
}

size_t local_size(int rank, data_layout &layout) {
    matrix_grid mat_grid(layout.matrix_dimension, layout.block_dimension);
    rank_grid_coord rank_coord =
        rank_to_grid(rank, layout.rank_grid, layout.rank_grid_ordering);
    local_blocks loc_blocks =
        get_local_blocks(mat_grid, layout.rank_grid, rank_coord);

    return loc_blocks.size_with_padding();
}
} // namespace scalapack
} // namespace grid2grid
