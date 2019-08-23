#include <grid2grid/cantor_mapping.hpp>
#include <grid2grid/transform.hpp>

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace grid2grid;
    using namespace grid2grid::scalapack;

    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (P != 4) {
        std::cout << "Runs with only 4 process.\n";
        std::abort();
    }

    matrix_dim m_dim{113, 56};         // global matrix size
    block_dim b_dim{27, 13};           // block dimension
    elem_grid_coord ij{20 + 1, 7 + 1}; // start of submatrix
    matrix_dim subm_dim{76, 37};       // dim of submatrix
    rank_decomposition r_grid{2, 2};
    scalapack::rank_grid_coord rank_src{0, 0};
    auto rank_grid_ordering = ordering::row_major;
    char transpose = 'N';

    // The buffer is largely irrelevant for this test.
    //
    int lld_m_dim = 60; // local leading dim
    std::vector<double> buffer(lld_m_dim * lld_m_dim);
    double *ptr = buffer.data();

    auto act_grid = get_scalapack_grid(lld_m_dim,
                                       m_dim,
                                       ij,
                                       subm_dim,
                                       b_dim,
                                       r_grid,
                                       rank_grid_ordering,
                                       transpose,
                                       rank_src,
                                       ptr,
                                       rank);

    // clang-format off
    std::vector<int> rows_split{0, 7, 34, 61, 76};
    std::vector<int> cols_split{0, 6, 19, 32, 37};
    std::vector<std::vector<int>> owners = {{0, 1, 0, 1},
                                            {2, 3, 2, 3},
                                            {0, 1, 0, 1},
                                            {2, 3, 2, 3}};
    // rows, cols, blk_coords, loc_ptr, stride
    std::vector<block<double>> loc_blks;
    if(rank == 0) {
        loc_blks = {
            {{ 0,  7}, { 0,  6}, {0, 0}, ptr + 20 +  7 * lld_m_dim, lld_m_dim},
            {{34, 61}, { 0,  6}, {2, 0}, ptr + 27 +  7 * lld_m_dim, lld_m_dim},
            {{ 0,  7}, {19, 32}, {0, 2}, ptr + 20 + 13 * lld_m_dim, lld_m_dim},
            {{34, 61}, {19, 32}, {2, 2}, ptr + 27 + 13 * lld_m_dim, lld_m_dim},
        };
    } else if(rank == 1) {
        loc_blks = {
            {{ 0,  7},  {6, 19}, {0, 1}, ptr + 20 +  0 * lld_m_dim, lld_m_dim},
            {{34, 61},  {6, 19}, {2, 1}, ptr + 27 +  0 * lld_m_dim, lld_m_dim},
            {{ 0,  7}, {32, 37}, {0, 3}, ptr + 20 + 13 * lld_m_dim, lld_m_dim},
            {{34, 61}, {32, 37}, {2, 3}, ptr + 27 + 13 * lld_m_dim, lld_m_dim},
        };
    } else if(rank == 2) {
        loc_blks = {
            {{ 7, 34}, { 0,  6}, {1, 0}, ptr +  0 +  7 * lld_m_dim, lld_m_dim},
            {{61, 76}, { 0,  6}, {3, 0}, ptr + 27 +  7 * lld_m_dim, lld_m_dim},
            {{ 7, 34}, {19, 32}, {1, 2}, ptr +  0 + 13 * lld_m_dim, lld_m_dim},
            {{61, 76}, {19, 32}, {3, 2}, ptr + 27 + 13 * lld_m_dim, lld_m_dim},
        };
    } else if(rank ==3) {
        loc_blks = {
            {{ 7, 34}, { 6, 19}, {1, 1}, ptr +  0 +  0 * lld_m_dim, lld_m_dim},
            {{61, 76}, { 6, 19}, {3, 1}, ptr + 27 +  0 * lld_m_dim, lld_m_dim},
            {{ 7, 34}, {32, 37}, {1, 3}, ptr +  0 + 13 * lld_m_dim, lld_m_dim},
            {{61, 76}, {32, 37}, {3, 3}, ptr + 27 + 13 * lld_m_dim, lld_m_dim},
        };
    }

    grid2grid::grid_layout<double> exp_grid{
        {
            {
                std::move(rows_split),
               std::move(cols_split)
            },
            std::move(owners), P
        },
        {
            std::move(loc_blks)
        }
    };
    // clang-format on

    bool OK =
        exp_grid.grid == act_grid.grid && exp_grid.blocks == act_grid.blocks;

    if (OK) {
        std::cout << "SUCCESS!\n";
    } else {
        std::cout << "FAIL!\n";
    }
}
