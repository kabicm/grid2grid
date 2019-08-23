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

    if (P > 1) {
        std::cout << "Runs with only 1 process.\n";
        std::abort();
    }

    int lld_m_dim = 200;                // local leading dim
    matrix_dim m_dim{200, 200};         // global matrix size
    elem_grid_coord ij{0 + 1, 100 + 1}; // start of submatrix
    matrix_dim subm_dim{100, 100};      // dim of submatrix
    block_dim b_dim{32, 32};            // block dimension
    rank_decomposition r_grid{1, 1};
    auto rank_grid_ordering = ordering::row_major;
    char transpose = 'N';
    scalapack::rank_grid_coord rank_src{0, 0};
    std::vector<double> buffer(200 * 200);
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
    std::vector<int> rows_split{0, 32, 64, 96, 100};
    std::vector<int> cols_split{0, 28, 60, 92, 100};
    std::vector<std::vector<int>> owners = {{0, 0, 0, 0},
                                            {0, 0, 0, 0},
                                            {0, 0, 0, 0},
                                            {0, 0, 0, 0}};
    std::vector<block<double>> loc_blks{
        // rows, cols, proc coords, loc_ptr, stride
        {{0,  32 }, {0, 28}, {0, 0}, ptr + 0  + 200 * 100, 200},
        {{32, 64 }, {0, 28}, {1, 0}, ptr + 32 + 200 * 100, 200},
        {{64, 96 }, {0, 28}, {2, 0}, ptr + 64 + 200 * 100, 200},
        {{96, 100}, {0, 28}, {3, 0}, ptr + 96 + 200 * 100, 200},

        {{0,  32 }, {28, 60}, {0, 1}, ptr + 0  + 200 * 128, 200},
        {{32, 64 }, {28, 60}, {1, 1}, ptr + 32 + 200 * 128, 200},
        {{64, 96 }, {28, 60}, {2, 1}, ptr + 64 + 200 * 128, 200},
        {{96, 100}, {28, 60}, {3, 1}, ptr + 96 + 200 * 128, 200},

        {{0,  32 }, {60, 92}, {0, 2}, ptr + 0  + 200 * 160, 200},
        {{32, 64 }, {60, 92}, {1, 2}, ptr + 32 + 200 * 160, 200},
        {{64, 96 }, {60, 92}, {2, 2}, ptr + 64 + 200 * 160, 200},
        {{96, 100}, {60, 92}, {3, 2}, ptr + 96 + 200 * 160, 200},

        {{0,  32 }, {92, 100}, {0, 3}, ptr + 0  + 200 * 192, 200},
        {{32, 64 }, {92, 100}, {1, 3}, ptr + 32 + 200 * 192, 200},
        {{64, 96 }, {92, 100}, {2, 3}, ptr + 64 + 200 * 192, 200},
        {{96, 100}, {92, 100}, {3, 3}, ptr + 96 + 200 * 192, 200},
    };
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
