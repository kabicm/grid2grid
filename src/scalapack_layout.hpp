#pragma once
#include <vector>
#include <tuple>
#include <iostream>
#include <cmath>

namespace grid2grid {
namespace scalapack {

enum ordering {
    row_major, column_major
};

struct rank_grid_coord {
    int row = 0;
    int col = 0;

    rank_grid_coord() = default;
    rank_grid_coord(int row, int col): row(row), col(col) {}
};

struct rank_decomposition {
    int n_rows = 0;
    int n_cols = 0;
    int n_total = 0;

    rank_decomposition() = default;
    rank_decomposition(int n_rows, int n_cols): n_rows(n_rows),
                                           n_cols(n_cols),
                                           n_total(n_rows*n_cols)
    {}
};

struct elem_grid_coord {
    int row = 0;
    int col = 0;

    elem_grid_coord() = default;
    elem_grid_coord(int row, int col): row(row), col(col) {}
};

struct local_grid_coord {
    elem_grid_coord el_coord;
    rank_grid_coord rank_coord;

    local_grid_coord() = default;
    local_grid_coord(elem_grid_coord el_coord, rank_grid_coord rank_coord):
        el_coord(el_coord), rank_coord(rank_coord) {}
};

struct block_dim {
    int n_rows = 0;
    int n_cols = 0;
    int size = 0;

    block_dim() = default;
    block_dim(int n_rows, int n_cols): n_rows(n_rows),
                                       n_cols(n_cols),
                                       size(n_rows*n_cols)
    {}
};

struct matrix_dim {
    int n_rows = 0;
    int n_cols = 0;
    int size = 0;

    matrix_dim() = default;
    matrix_dim(int n_rows, int n_cols): n_rows(n_rows),
                                        n_cols(n_cols),
                                        size(n_rows*n_cols)
    {}
};

struct matrix_grid {
    matrix_dim matrix_dimension;
    block_dim block_dimension;

    matrix_grid() = default;
    matrix_grid(matrix_dim mat_dim, block_dim b_dim):
        matrix_dimension(mat_dim), block_dimension(b_dim) {}
};

struct local_blocks {
    int n_blocks_row = 0;
    int n_blocks_col = 0;

    block_dim block_dimension;
    rank_grid_coord rank_coord;
    int stride = 0;

    size_t size_no_padding = 0;

    local_blocks() = default;
    local_blocks(int n_bl_row, int n_bl_col, block_dim b_dim, rank_grid_coord r_coord):
            n_blocks_row(n_bl_row), n_blocks_col(n_bl_col),
            block_dimension(b_dim), rank_coord(r_coord) {
        size_no_padding = n_blocks_row * n_blocks_col * b_dim.size;
        stride = n_blocks_row * b_dim.n_rows;
    }

    local_grid_coord get_local_coordinates(int index) {
        int row = index % stride;
        int col = index / stride;

        return {{row, col}, rank_coord};
    }

    size_t size_with_padding() {
        return n_blocks_row * n_blocks_col * block_dimension.size;
    }
};

struct data_layout {
    matrix_dim matrix_dimension;
    block_dim block_dimension;
    rank_decomposition rank_grid;
    ordering rank_grid_ordering = ordering::column_major;

    data_layout() = default;

    data_layout(matrix_dim m_dim, block_dim b_dim,
            rank_decomposition r_grid, ordering r_ordering):
        matrix_dimension(m_dim), block_dimension(b_dim),
        rank_grid(r_grid), rank_grid_ordering(r_ordering) 
    {}
};

rank_grid_coord rank_to_grid(int rank, rank_decomposition grid_dim, ordering grid_ord);

int rank_from_grid(rank_grid_coord grid_coord, rank_decomposition grid_dim, ordering grid_ord);

std::tuple<int, int> local_coordinate(int glob_coord, int block_dimension,
                                      int p_block_dimension, int mat_dim);

// global->local coordinates
local_grid_coord local_coordinates(matrix_grid mat_grid, rank_decomposition rank_grid,
                                   elem_grid_coord global_coord);

// local->global coordinates
elem_grid_coord global_coordinates(matrix_grid mat_grid, rank_decomposition rank_grid,
                                   local_grid_coord local_coord);

local_blocks get_local_blocks(matrix_grid mat_grid, rank_decomposition r_grid,
        rank_grid_coord rank_coord);

size_t local_size(int rank, data_layout& layout);

template <typename T, typename Function>
void initialize_locally(T* buffer, Function f, int rank, data_layout& layout) {
    // using elem_type = decltype(f(0, 0));
    using elem_type = T;

    matrix_grid mat_grid(layout.matrix_dimension, layout.block_dimension);
    rank_grid_coord rank_coord = rank_to_grid(rank, layout.rank_grid, layout.rank_grid_ordering);
    local_blocks loc_blocks = get_local_blocks(mat_grid, layout.rank_grid, rank_coord);
    size_t buffer_size = loc_blocks.size_with_padding();

    for (size_t i = 0; i < buffer_size; ++i) {
        local_grid_coord local_coord = loc_blocks.get_local_coordinates(i);
        elem_grid_coord global_coord = global_coordinates(mat_grid, layout.rank_grid, local_coord);
        if (global_coord.row > -1 && global_coord.col > -1) {
            *(buffer + i) = (elem_type) f(global_coord.row, global_coord.col);
        } else {
            *(buffer + i) = elem_type();
        }
    }

#ifdef DEBUG
    std::cout << "Initializing the buffer of size: " << buffer_size << std::endl;
    std::cout << "elements = ";
    for (unsigned i = 0; i < buffer_size; ++i) {
        std::cout << *(buffer+i) << ", ";
    }
    std::cout << std::endl;
#endif
}

template <typename Function>
bool validate(Function f, std::vector<decltype(f(0, 0))>& buffer, int rank, data_layout& layout, double eps = 1e-6) {
    using elem_type = decltype(f(0, 0));
    matrix_grid mat_grid(layout.matrix_dimension, layout.block_dimension);
    rank_grid_coord rank_coord = rank_to_grid(rank, layout.rank_grid, layout.rank_grid_ordering);
    local_blocks loc_blocks = get_local_blocks(mat_grid, layout.rank_grid, rank_coord);

    bool correct = true;

    if (buffer.size() != loc_blocks.size_with_padding()) {
        std::cout << "Size of the buffer is " << 
            buffer.size() << " and should be " << loc_blocks.size_with_padding() << std::endl;
        correct = false;
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
        local_grid_coord local_coord = loc_blocks.get_local_coordinates(i);
        elem_grid_coord global_coord = global_coordinates(mat_grid, layout.rank_grid, local_coord);
        if (global_coord.row < 0 || global_coord.col < 0
                || global_coord.row >= layout.matrix_dimension.n_rows 
                || global_coord.col >= layout.matrix_dimension.n_cols) {
            continue;
        }
        auto target_value = (elem_type) f(global_coord.row, global_coord.col);
        if (std::abs(buffer[i] - target_value) > eps) {
            // auto coord = invert_cantor_pairing((int) buffer[i]);
            std::cout << "(" << global_coord.row << ", " << global_coord.col << ") = "
                << buffer[i] << " instead of " << target_value <<  std::endl;
            correct = false;
        }
        // else {
        //     std::cout << "(" << global_coord.row << ", " << global_coord.col << ") = is correct" <<  std::endl;
        // }
    }
#ifdef DEBUG
    if (correct) {
        std::cout << "Values are correct!" << std::endl;
    }
#endif
    return correct;
}
}}
