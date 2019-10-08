#include <grid2grid/cantor_mapping.hpp>
#include <grid2grid/transform.hpp>

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace grid2grid;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int P, rank;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    if (P != 4) {
        std::cout << "[ERROR] The test runs with 4 processes!\n";
        MPI_Abort(comm, 0);
    }

    auto values = [](int i, int j) {
        return 1.0 * grid2grid::cantor_pairing(i, j);
    };

    // initialize the local buffer as given by function values
    // function 'values': maps global coordinates of the matrix to values
    //    std::vector<double> buffer1(local_size(rank, layout1));
    //    initialize_locally(buffer1.data(), values, rank, layout1);

    double *ptr; // TODO: init properly

    std::vector<block<double>> init_layout_blocks;
    std::vector<block<double>> final_layout_blocks;

    // clang-format off
    //
    // rows, cols, proc coords, loc_ptr, stride
    //
    if (rank == 0) {
        init_layout_blocks = {
            {{ 0, 32}, {28,  60}, {0, 1}, ptr, 104},
            {{64, 96}, {28,  60}, {2, 1}, ptr, 104},
            {{ 0, 32}, {92, 100}, {0, 3}, ptr, 104},
            {{64, 96}, {92, 100}, {2, 3}, ptr, 104},
        };
        final_layout_blocks = {
            {{0, 100}, {0, 25}, {0, 0}, ptr, 100}
        };
    } else if (rank == 1) {
        init_layout_blocks = {
            {{32,  64}, {28,  60}, {1, 1}, ptr, 104},
            {{96, 100}, {28,  60}, {3, 1}, ptr, 104},
            {{32,  64}, {92, 100}, {1, 3}, ptr, 104},
            {{96, 100}, {92, 100}, {3, 3}, ptr, 104}
        };
        final_layout_blocks = {
            {{0, 100}, {25, 50}, {0, 1}, ptr, 100}
        };
    } else if (rank == 2) {
        init_layout_blocks = {
            {{ 0, 32}, { 0, 28}, {0, 0}, ptr, 104},
            {{64, 96}, { 0, 28}, {2, 0}, ptr, 104},
            {{ 0, 32}, {60, 92}, {0, 2}, ptr, 104},
            {{64, 96}, {60, 92}, {2, 2}, ptr, 104},
        };
        final_layout_blocks = {
            {{0, 100}, {50, 75}, {0, 2}, ptr, 100}
        };
    } else { // rank == 3
        init_layout_blocks = {
            {{32,  64}, { 0, 28}, {1, 0}, ptr, 104},
            {{96, 100}, { 0, 28}, {3, 0}, ptr, 104},
            {{32,  64}, {60, 92}, {1, 2}, ptr, 104},
            {{96, 100}, {60, 92}, {3, 2}, ptr, 104},
        };
        final_layout_blocks = {
            {{0, 100}, {75, 100}, {0, 3}, ptr, 100}
        };
    }

    grid2grid::grid_layout<double> init_layout{
        {
            {
                {0, 32, 64, 96, 100},
                {0, 28, 60, 92, 100}
            },
            {
                {2, 0, 2, 0},
                {3, 1, 3, 1},
                {2, 0, 2, 0},
                {3, 1, 3, 1}
            }, P
        },
        {
            std::move(init_layout_blocks)
        }
    };

    grid2grid::grid_layout<double> final_layout{
        {
            {
                {0, 25, 50, 75, 100},
                {0, 100}
            },
            {
                {0, 1, 2, 3}
            }, P
        },
        {
            std::move(final_layout_blocks)
        }
    };
    // clang-format on

    grid2grid::transform<double>(init_layout, final_layout, comm);

    // check if the values of buffer1 correspond to values
    // given by argument function 'values'
    bool ok = validate(values, buffer1, rank, layout1);

    //        initialize_locally(buffer2.data(), values, rank, layout2);

    // check if the values of buffer1 correspond to values
    // given by argument function 'values'
    //    ok = ok && validate(values, buffer2, rank, layout2);

    MPI_Finalize();
    //    return !ok;
}
