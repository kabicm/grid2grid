// STL
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// Local
#include <transform.hpp>
#include <math_utils.hpp>
#include <options.hpp>

using namespace grid2grid;

int main( int argc, char **argv ) {
    options::initialize(argc, argv);

    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto m = options::next_int("-m", "--rows", "number of rows.", 10);
    auto n = options::next_int("-n", "--cols", "number of rows.", 12);

    auto bm1 = options::next_int("-ibm", "--init_block_rows", "Initial block size for rows.", 2);
    auto bn1 = options::next_int("-ibn", "--init_block_cols", "Initial block size for columns.", 2);

    auto bm2 = options::next_int("-fbm", "--final_block_rows", "Final block size for rows.", 4);
    auto bn2 = options::next_int("-fbn", "--final_block_cols", "Final block size for columns.", 5);

    auto pm = options::next_int("-pm", "--p_rows", "Processor grid rows.", 2);
    auto pn = options::next_int("-pn", "--p_cols", "Processor grid column.", 2);

    if (rank == 0) {
        std::cout << "Matrix size = (" << m << ", " << n << ")" << std::endl;
        std::cout << "Initial block size = (" << bm1 << ", " << bn1 << ")" << std::endl;
        std::cout << "Final block size = (" << bm2 << ", " << bn2 << ")" << std::endl;
        std::cout << "Processor grid = (" << pm << ", " << pn << ")" << std::endl;
    }

    auto values = [](int i, int j) {
        return 1.0 * cantor_pairing(i, j);
    };

    auto zeros = [](int i, int j) {
        return 0.0;
    };

    scalapack::ordering ordering = scalapack::ordering::column_major;

    scalapack::data_layout layout1({m, n}, {bm1, bn1}, {pm, pn}, ordering);
    auto buffer1 = initialize_locally(values, rank, layout1);
    bool ok = validate(values, buffer1, rank, layout1);

    MPI_Barrier(MPI_COMM_WORLD);

    grid_layout<double> scalapack_layout_1 = get_scalapack_grid(layout1, buffer1.data(), rank);

    scalapack::data_layout layout2({m, n}, {bm2, bn2}, {pm, pn}, ordering);
    auto buffer2 = initialize_locally(zeros, rank, layout2);
    grid_layout<double> scalapack_layout_2 = get_scalapack_grid(layout2, buffer2.data(), rank);

    transform(scalapack_layout_1, scalapack_layout_2, MPI_COMM_WORLD);

    ok = ok && validate(values, buffer2, rank, layout2);

    std::cout << "Rank " << rank << ": result is" << (ok ? "" : " not") << " correct!" << std::endl;;

    MPI_Finalize();

    return !ok;
}
