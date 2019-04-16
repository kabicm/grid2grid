#include "scalapack_layout.hpp"
namespace grid2grid {
namespace scalapack {

rank_grid_coord rank_to_grid(int rank, rank_decomposition grid_dim, ordering grid_ord) {
    if (rank < 0 || rank >= grid_dim.n_total) {
        throw std::runtime_error("Error in rank_to_grid: rank does not belong to the grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.n_rows;
        return {rank % ld, rank / ld};
    } else {
        int ld = grid_dim.n_cols;
        return {rank / ld, rank % ld};
    }
}

int rank_from_grid(rank_grid_coord grid_coord, rank_decomposition grid_dim, ordering grid_ord) {
    if (grid_coord.row < 0 || grid_coord.row >= grid_dim.n_rows
        || grid_coord.col < 0 || grid_coord.col >= grid_dim.n_cols) {
        throw std::runtime_error("Error in rank_from_grid: rank coordinates do not belong \
    to the rank grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.n_rows;
        return grid_coord.col * ld + grid_coord.row;
    } else {
        int ld = grid_dim.n_cols;
        return grid_coord.row * ld + grid_coord.col;
    }
}

std::tuple<int, int> local_coordinate(int glob_coord, int block_dimension,
                                      int p_block_dimension, int mat_dim) {
    int idx_block = glob_coord / block_dimension;
    int idx_in_block = glob_coord % block_dimension;
    int idx_block_proc = idx_block / p_block_dimension;
    int owner = idx_block % p_block_dimension;

    return {idx_block_proc * block_dimension + idx_in_block, owner};
}

// global->local coordinates
local_grid_coord local_coordinates(matrix_grid mat_grid, rank_decomposition rank_grid,
                                   elem_grid_coord global_coord) {
    int row, p_row;
    std::tie<int, int>(row, p_row) = local_coordinate(global_coord.row,
                                                      mat_grid.block_dimension.n_rows, rank_grid.n_rows,
                                                      mat_grid.matrix_dimension.n_rows);
    int col, p_col;
    std::tie<int, int>(col, p_col) = local_coordinate(global_coord.col,
                                                      mat_grid.block_dimension.n_cols, rank_grid.n_cols,
                                                      mat_grid.matrix_dimension.n_rows);
    return {{row, col}, {p_row, p_col}};
}

// local->global coordinates
elem_grid_coord global_coordinates(matrix_grid mat_grid, rank_decomposition rank_grid,
                                   local_grid_coord local_coord) {
    int li = local_coord.el_coord.row;
    int lj = local_coord.el_coord.col;

    int my_row_rank = local_coord.rank_coord.row;
    int my_col_rank = local_coord.rank_coord.col;

    int num_row_ranks = rank_grid.n_rows;
    int num_col_ranks = rank_grid.n_cols;

    int mb = mat_grid.block_dimension.n_rows;
    int nb = mat_grid.block_dimension.n_cols;

    int gi = ((li / mb) * num_row_ranks + my_row_rank) * mb + (li % mb);
    int gj = ((lj / nb) * num_col_ranks + my_col_rank) * nb + (lj % nb);

    if (gi < 0 || gi > mat_grid.matrix_dimension.n_rows
        || gj < 0 || gj >= mat_grid.matrix_dimension.n_cols) {
        // std::cout << "coordinates (" << gi << ", " << gj << ") should belong to ("
        //     << mat_grid.matrix_dimension.n_rows << ", " << mat_grid.matrix_dimension.n_cols << ")" << std::endl;
        // throw std::runtime_error("ERROR in scalapack::global_coordinates, values out of range.");
        return {-1, -1};
    }
    return {gi, gj};
}

local_blocks get_local_blocks(matrix_grid mat_grid, rank_decomposition r_grid,
                              rank_grid_coord rank_coord) {
    auto b_dim = mat_grid.block_dimension;
    auto m_dim = mat_grid.matrix_dimension;

    int n_blocks_row = (int) std::ceil(1.0 * m_dim.n_rows / b_dim.n_rows);
    int n_blocks_col = (int) std::ceil(1.0 * m_dim.n_cols / b_dim.n_cols);

    int n_owning_blocks_row = n_blocks_row / r_grid.n_rows
                              + (rank_coord.row < n_blocks_row % r_grid.n_rows ? 1 : 0);
    int n_owning_blocks_col = n_blocks_col / r_grid.n_cols
                              + (rank_coord.col < n_blocks_col % r_grid.n_cols ? 1 : 0);

    return {n_owning_blocks_row, n_owning_blocks_col, b_dim, rank_coord};
}
}}
