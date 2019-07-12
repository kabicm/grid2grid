#include <transform.hpp>

#include <complex>

namespace grid2grid {

template <typename T>
std::vector<message<T>> decompose_block(const block<T> &b,
                                        grid_cover &g_cover,
                                        const assigned_grid2D &g) {
    // std::cout << "decomposing block " << b << std::endl;
    block_cover b_cover = g_cover.decompose_block(b);

    int row_first = b_cover.rows_cover.start_index;
    int row_last = b_cover.rows_cover.end_index;

    int col_first = b_cover.cols_cover.start_index;
    int col_last = b_cover.cols_cover.end_index;

    std::vector<message<T>> decomposed_blocks;

    int row_start = b.rows_interval.start;
    // use start of the interval to get the rank and the end of the interval
    // to get the block which has to be sent
    // skip the last element
    for (int i = row_first; i < row_last; ++i) {
        int row_end = std::min(g.grid().rows_split[i + 1], b.rows_interval.end);

        int col_start = b.cols_interval.start;
        for (int j = col_first; j < col_last; ++j) {
            // use i, j to find out the rank
            int rank = g.owner(i, j);
            // std::cout << "owner of block " << i << ", " << j << " is " <<
            // rank << std::endl;

            // use i+1 and j+1 to find out the block
            int col_end =
                std::min(g.grid().cols_split[j + 1], b.cols_interval.end);

            // get pointer to this block of data based on the internal local
            // layout
            block<T> subblock =
                b.subblock({row_start, row_end}, {col_start, col_end});

            assert(subblock.non_empty());
            // if non empty, add this block
            if (subblock.non_empty()) {
                // std::cout << "for rank " << rank << ", adding subblock: " <<
                // subblock << std::endl; std::cout << "owner of " << subblock
                // << " is " << rank << std::endl;
                decomposed_blocks.push_back({subblock, rank});
            }

            col_start = col_end;
        }
        row_start = row_end;
    }
    return decomposed_blocks;
}

template <typename T>
void merge_messages(std::vector<message<T>> &messages) {
    std::sort(messages.begin(), messages.end());
}

template <typename T>
std::vector<message<T>> decompose_blocks(const grid_layout<T> &init_layout,
                                         const grid_layout<T> &final_layout) {
    grid_cover g_overlap(init_layout.grid.grid(), final_layout.grid.grid());

    std::vector<message<T>> messages;

    for (int i = 0; i < init_layout.blocks.num_blocks(); ++i) {
        // std::cout << "decomposing block " << i << " out of " <<
        // init_layout.blocks.num_blocks() << std::endl;
        auto blk = init_layout.blocks.get_block(i);
        assert(blk.non_empty());
        std::vector<message<T>> decomposed =
            decompose_block(blk, g_overlap, final_layout.grid);
        messages.insert(messages.end(), decomposed.begin(), decomposed.end());
    }
    merge_messages(messages);
    return messages;
}

template <typename T>
communication_data<T> prepare_to_send(const grid_layout<T> &init_layout,
                                      const grid_layout<T> &final_layout) {
    std::vector<message<T>> messages =
        decompose_blocks(init_layout, final_layout);
    return communication_data<T>(std::move(messages), final_layout.num_ranks());
}

template <typename T>
communication_data<T> prepare_to_recv(const grid_layout<T> &final_layout,
                                      const grid_layout<T> &init_layout) {
    std::vector<message<T>> messages =
        decompose_blocks(final_layout, init_layout);
    return communication_data<T>(std::move(messages), init_layout.num_ranks());
}

inline std::vector<int>
line_split(scalapack::elem_grid_coord initial_coord,
           scalapack::elem_grid_coord final_coord,
           scalapack::block_dim b_dim,
           int index) { // index == 0 => rows, index == 1 => cols
    std::vector<int> splits;
    auto m_dim_offset = final_coord - initial_coord;
    scalapack::elem_grid_coord ij_next = initial_coord;

    while (ij_next[index] <= final_coord[index]) {
        ij_next[index] = (ij_next[index] / b_dim[index]) * b_dim[index];
        auto ij_next_offset = ij_next[index] - initial_coord[index];
        splits.push_back(std::min(ij_next_offset, m_dim_offset[index]));

        ij_next[index] += b_dim[index];
    }

    if (splits.back() != m_dim_offset[index]) {
        splits.push_back(m_dim_offset[index]);
    }
    return splits;
}

template <typename T>
grid_layout<T>
get_scalapack_grid(int lld,                        // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   T *ptr,
                   const int rank) {
    if (transposed) {
        m_dim.transpose();
        ij.transpose();
    }

    // **************************
    // create grid2D
    // **************************
    // prepare row intervals
    // and col intervals
    scalapack::elem_grid_coord one{1, 1};
    ij = ij - one;
    scalapack::elem_grid_coord ij_init = ij;

    scalapack::elem_grid_coord submatrix_end =
        (scalapack::elem_grid_coord)(ij_init + subm_dim);

    std::vector<int> rows_split = line_split(ij_init, submatrix_end, b_dim, 0);
    std::vector<int> cols_split = line_split(ij_init, submatrix_end, b_dim, 1);

#ifdef DEBUG
    std::cout << "rows_split = " << std::endl;
    for (const auto &el : rows_split) {
        std::cout << el << ", " << std::endl;
    }
    std::cout << "cols_split = " << std::endl;
    for (const auto &el : cols_split) {
        std::cout << el << ", " << std::endl;
    }
#endif

    int n_blocks_row = rows_split.size() - 1;
    int n_blocks_col = cols_split.size() - 1;

    grid2D grid(std::move(rows_split), std::move(cols_split));

    // **************************
    // create an assigned grid2D
    // **************************
    // create a matrix of ranks owning each block
    std::vector<std::vector<int>> owners(n_blocks_row,
                                         std::vector<int>(n_blocks_col));
    for (int i = 0; i < n_blocks_row; ++i) {
        int rank_row = i % r_grid.row;
        for (int j = 0; j < n_blocks_col; ++j) {
            int rank_col = j % r_grid.col;
            owners[i][j] = rank_from_grid(
                {rank_row, rank_col}, r_grid, rank_grid_ordering, rank_src);
        }
    }

    // create an assigned grid2D
    assigned_grid2D assigned_grid(
        std::move(grid), std::move(owners), r_grid.n_total());

    // **************************
    // create local memory view
    // **************************
    // get coordinates of current rank in a rank decomposition
    scalapack::rank_grid_coord rank_coord =
        rank_to_grid(rank, r_grid, rank_grid_ordering, rank_src);

    std::vector<block<T>> loc_blocks;

    int stride = lld;

    block_range submatrix({ij_init.row, submatrix_end.row},
                          {ij_init.col, submatrix_end.col});

    auto bij_init = ij / b_dim;
    auto remainder = bij_init % r_grid;
    bij_init = bij_init / r_grid;

    if (rank_coord[0] < remainder[0]) {
        bij_init[0]++;
    }
    if (rank_coord[1] < remainder[1]) {
        bij_init[1]++;
    }

    // iterate through all the blocks that this rank owns
    for (int bj = rank_coord.col; bj < n_blocks_col; bj += r_grid.col) {
        interval col_interval = assigned_grid.cols_interval(bj);

        for (int bi = rank_coord.row; bi < n_blocks_row; bi += r_grid.row) {
            interval row_interval = assigned_grid.rows_interval(bi);

            block_range current_block(row_interval, col_interval);
            if (current_block.outside_of(submatrix))
                continue;

            int offset = 0;

            auto intersection = current_block.intersection(submatrix);
            if (intersection != current_block) {
                int row_within_block = std::abs(submatrix.rows_interval.start -
                                                row_interval.start);
                int col_within_block = std::abs(submatrix.cols_interval.start -
                                                col_interval.start);
                offset = lld * col_within_block + row_within_block;
            }

            auto data = ptr + offset + (bj / r_grid.col) * stride * b_dim.col +
                        (bi / r_grid.row) * b_dim.row;
            assert(data != nullptr);
            // if (data == nullptr) {
            //     std::cout << "Data = nullptr for bi = " << bi << " and bj = "
            //     << bj << std::endl;
            // }
            // std::cout << "scalapack" << std::endl;
            // std::cout << "row_interval = " << row_interval << ", col_interval
            // " << col_interval << std::endl; std::cout << "ptr offset = " <<
            // (data - ptr) << std::endl;
            int bi_shifted = bi - bij_init[0];
            int bj_shifted = bj - bij_init[1];

            block<T> blk;

            if (offset == 0) {
                blk = block<T>{
                    assigned_grid, {bi_shifted, bj_shifted}, data, stride};
            } else {
                block_range b(row_interval, col_interval);
                block_range b_overlap = b.intersection(submatrix);
                assert(b_overlap.non_empty());
                blk =
                    block<T>{b_overlap, {bi_shifted, bj_shifted}, data, stride};
            }
            assert(blk.non_empty());
            loc_blocks.push_back(blk);
        }
    }

    local_blocks<T> local_memory(std::move(loc_blocks));
    // std::cout << "local blocks:" << local_memory << std::endl;

    // **************************
    // create a grid layout
    // **************************
    return {std::move(assigned_grid), std::move(local_memory)};
}

template <typename T>
grid_layout<T>
get_scalapack_grid(int lld,                        // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const T *ptr,
                   const int rank) {
    return get_scalapack_grid(lld,
                              m_dim,
                              ij,
                              subm_dim,
                              b_dim,
                              r_grid,
                              rank_grid_ordering,
                              transposed,
                              rank_src,
                              const_cast<T *>(ptr),
                              rank);
}

template <typename T>
grid_layout<T> get_scalapack_grid(scalapack::matrix_dim m_dim,
                                  scalapack::block_dim b_dim,
                                  scalapack::rank_decomposition r_grid,
                                  scalapack::ordering rank_grid_ordering,
                                  T *ptr,
                                  int rank) {
    // std::cout << "I AM RANK " << rank << std::endl;
    int n_blocks_row = (int)std::ceil(1.0 * m_dim.row / b_dim.row);
    int n_blocks_col = (int)std::ceil(1.0 * m_dim.col / b_dim.col);

    scalapack::rank_grid_coord rank_coord =
        rank_to_grid(rank, r_grid, rank_grid_ordering);

    int n_owning_blocks_row =
        n_blocks_row / r_grid.row +
        (rank_coord.row < (n_blocks_row % r_grid.row) ? 1 : 0);

    int stride = n_owning_blocks_row * b_dim.row;

    bool transposed = false;

    return get_scalapack_grid(stride, // local leading dim
                              m_dim,  // global matrix size
                              {1, 1}, // start of submatrix
                              m_dim,  // dim of submatrix
                              b_dim,  // block dimension
                              r_grid,
                              rank_grid_ordering,
                              transposed,
                              {0, 0},
                              ptr,
                              rank);
}

template <typename T>
grid_layout<T>
get_scalapack_grid(scalapack::data_layout &layout, T *ptr, int rank) {
    return get_scalapack_grid<T>(layout.matrix_dimension,
                                 layout.block_dimension,
                                 layout.rank_grid,
                                 layout.rank_grid_ordering,
                                 ptr,
                                 rank);
}

template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm) {
    // MPI_Barrier(comm);
    // auto total_start = std::chrono::steady_clock::now();
    communication_data<T> send_data =
        prepare_to_send(initial_layout, final_layout);
    // auto start = std::chrono::steady_clock::now();
    // auto prepare_send =
    // std::chrono::duration_cast<std::chrono::milliseconds>(start -
    // total_start).count();
    communication_data<T> recv_data =
        prepare_to_recv(final_layout, initial_layout);
    // auto end = std::chrono::steady_clock::now();
    // auto prepare_recv =
    // std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count();
    int rank;
    MPI_Comm_rank(comm, &rank);

    // copy blocks to temporary send buffers
    // start = std::chrono::steady_clock::now();
    send_data.copy_to_buffer();
    // end = std::chrono::steady_clock::now();
    // auto copy_to_buffer_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count();
#ifdef DEBUG
    std::cout << "send buffer content: " << std::endl;
    for (int i = 0; i < send_data.total_size; ++i) {
        // std::pair<int, int> el =
        // math_utils::invert_cantor_pairing((int)send_data.buffer[i]);
        std::cout << send_data.buffer[i] << ", ";
    }
    std::cout << std::endl;
#endif
    // start = std::chrono::steady_clock::now();
    // perform the communication
    MPI_Alltoallv(send_data.data(),
                  send_data.counts.data(),
                  send_data.dspls.data(),
                  mpi_type_wrapper<T>::type(),
                  recv_data.data(),
                  recv_data.counts.data(),
                  recv_data.dspls.data(),
                  mpi_type_wrapper<T>::type(),
                  comm);
    // end = std::chrono::steady_clock::now();
    // auto comm_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count();

#ifdef DEBUG
    std::cout << "recv buffer content: " << std::endl;
    for (int i = 0; i < recv_data.total_size; ++i) {
        // std::pair<int, int> el =
        // math_utils::invert_cantor_pairing((int)recv_data.buffer[i]);
        std::cout << recv_data.buffer[i] << ", ";
    }
    std::cout << std::endl;
#endif
    // start = std::chrono::steady_clock::now();
    // copy blocks from a temporary buffer back to blocks
    recv_data.copy_from_buffer();
    // end = std::chrono::steady_clock::now();
    // auto copy_from_buffer_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count();

    // auto total_end = std::chrono::steady_clock::now();
    // auto total_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(total_end -
    // total_start).count(); if (rank == 0) {
    //     std::cout << "prepare send: " << prepare_send << std::endl;
    //     std::cout << "prepare recv: " << prepare_recv << std::endl;
    //     std::cout << "copy: blocks -> buffer: " << copy_to_buffer_duration <<
    //     std::endl; std::cout << "communication: : " << comm_duration <<
    //     std::endl; std::cout << "copy: buffer -> blocks: " <<
    //     copy_from_buffer_duration << std::endl; std::cout << "total: " <<
    //     total_duration << std::endl;
    // }
}

template void transform<float>(grid_layout<float> &initial_layout,
                               grid_layout<float> &final_layout,
                               MPI_Comm comm);

template void transform<double>(grid_layout<double> &initial_layout,
                                grid_layout<double> &final_layout,
                                MPI_Comm comm);

template void
transform<std::complex<float>>(grid_layout<std::complex<float>> &initial_layout,
                               grid_layout<std::complex<float>> &final_layout,
                               MPI_Comm comm);

template void transform<std::complex<double>>(
    grid_layout<std::complex<double>> &initial_layout,
    grid_layout<std::complex<double>> &final_layout,
    MPI_Comm comm);

template grid_layout<float>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   float *ptr,
                   const int rank);

template grid_layout<double>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   double *ptr,
                   const int rank);

template grid_layout<std::complex<float>>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   std::complex<float> *ptr,
                   const int rank);

template grid_layout<std::complex<double>>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   std::complex<double> *ptr,
                   const int rank);

template grid_layout<float>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const float *ptr,
                   const int rank);

template grid_layout<double>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const double *ptr,
                   const int rank);

template grid_layout<std::complex<float>>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const std::complex<float> *ptr,
                   const int rank);

template grid_layout<std::complex<double>>
get_scalapack_grid(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const std::complex<double> *ptr,
                   const int rank);

template grid_layout<float>
get_scalapack_grid(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   float *ptr,
                   int rank);

template grid_layout<double>
get_scalapack_grid(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   double *ptr,
                   int rank);

template grid_layout<std::complex<float>>
get_scalapack_grid(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   std::complex<float> *ptr,
                   int rank);

template grid_layout<std::complex<double>>
get_scalapack_grid(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   std::complex<double> *ptr,
                   int rank);

template grid_layout<float>
get_scalapack_grid(scalapack::data_layout &layout, float *ptr, int rank);

template grid_layout<double>
get_scalapack_grid(scalapack::data_layout &layout, double *ptr, int rank);

template grid_layout<std::complex<float>>
get_scalapack_grid(scalapack::data_layout &layout,
                   std::complex<float> *ptr,
                   int rank);

template grid_layout<std::complex<double>>
get_scalapack_grid(scalapack::data_layout &layout,
                   std::complex<double> *ptr,
                   int rank);

} // namespace grid2grid
