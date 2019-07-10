#include "grid2D.hpp"
namespace grid2grid {
/*
    A class describing a matrix n_rows x n_cols split into an arbitrary grid.
    A grid is defined by rows_split and cols_split. More precisely, a matrix is
   split into blocks such that a block with coordinates (i, j) is defined by:
        - row interval [rows_split[i], rows_split[i+1]) and similarly
        - column interval [cols_split[i], cols_split[i+1])
    For each i, the following must hold:
        - 0 <= rows_split[i] < n_rows
        - 0 <= cols_split[i] < n_cols
*/
grid2D::grid2D(std::vector<int> &&r_split, std::vector<int> &&c_split)
    : n_rows(r_split.size() - 1)
    , n_cols(c_split.size() - 1)
    , rows_split(std::forward<std::vector<int>>(r_split))
    , cols_split(std::forward<std::vector<int>>(c_split)) {}

// returns index-th row interval, i.e. [rows_split[index], rows_split[index +
// 1])
interval grid2D::row_interval(int index) const {
    if ((size_t)index >= rows_split.size() - 1) {
        throw std::runtime_error(
            "ERROR: in class grid2D, row index out of range.");
    }
    return {rows_split[index], rows_split[index + 1]};
}

// returns index-th column interval, i.e. [cols_split[index], cols_split[index +
// 1])
interval grid2D::col_interval(int index) const {
    if ((size_t)index >= cols_split.size() - 1) {
        throw std::runtime_error(
            "ERROR: in class grid2D, col index out of range.");
    }
    return {cols_split[index], cols_split[index + 1]};
}

/*
A class describing a matrix split into an arbitrary grid (grid2D) where each
block is assigned to an arbitrary MPI rank. More precisely, a block with
coordinates (i, j) is assigned to an MPI rank defined by ranks[i][j].
*/
assigned_grid2D::assigned_grid2D(grid2D &&g,
                                 std::vector<std::vector<int>> &&proc,
                                 int n_ranks)
    : g(std::forward<grid2D>(g))
    , ranks(std::forward<std::vector<std::vector<int>>>(proc))
    , n_ranks(n_ranks) {}

// returns the rank owning block (i, j)
int assigned_grid2D::owner(int i, int j) const { return ranks[i][j]; }

// returns a grid
const grid2D &assigned_grid2D::grid() const { return g; }

// returns a total number of MPI ranks in the communicator
// which owns the whole matrix. However, not all ranks have
// to own a block of the matrix. Thus, it might happen that
// maximum(ranks[i][j]) < n_ranks - 1 over all i, j
int assigned_grid2D::num_ranks() const { return n_ranks; }

// returns index-th row interval
interval assigned_grid2D::rows_interval(int index) const {
    return g.row_interval(index);
}

// returns index-th column interval
interval assigned_grid2D::cols_interval(int index) const {
    return g.col_interval(index);
}

int assigned_grid2D::block_size(int row_index, int col_index) {
    return rows_interval(row_index).length() *
           cols_interval(col_index).length();
}
} // namespace grid2grid
