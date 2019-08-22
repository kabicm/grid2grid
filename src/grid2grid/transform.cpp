#include <grid2grid/transform.hpp>

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

inline std::vector<int> line_split(int begin, int end, int blk_len) {
    int len = end - begin;
    int rem = blk_len - begin % blk_len;

    std::vector<int> splits{0};

    if (rem >= len) {
        splits.push_back(len);
        return splits;
    }

    if (rem != 0) {
        splits.push_back(rem);
    }

    int num_blocks = (len - rem) / blk_len;
    for (int i = 0; i < num_blocks; ++i) {
        splits.push_back(splits.back() + blk_len);
    }

    if (splits.back() != len) {
        splits.push_back(len);
    }

    return splits;
}

template <typename T>
grid_layout<T> get_scalapack_grid(
    int lld,                                    // local leading dim
    scalapack::matrix_dim matrix_shape,         // global matrix size
    scalapack::elem_grid_coord submatrix_begin, // start of submatrix (from 1)
    scalapack::matrix_dim submatrix_shape,      // dim of submatrix
    scalapack::block_dim blk_shape,             // block dimension
    scalapack::rank_decomposition ranks_grid,
    scalapack::ordering ranks_grid_ordering,
    char transpose_flag,
    scalapack::rank_grid_coord matrix_rank_src,
    T *ptr,
    const int rank) {

    // ----------- MATRIX ----------------------
    std::vector<int> matrix_rows_split =
        line_split(0, matrix_shape[0], blk_shape[0]);
    std::vector<int> matrix_cols_split =
        line_split(0, matrix_shape[1], blk_shape[1]);

    int matrix_blk_grid_rows = static_cast<int>(matrix_rows_split.size() - 1);
    int matrix_blk_grid_cols = static_cast<int>(matrix_cols_split.size() - 1);

    std::vector<std::vector<int>> matrix_owners(
        matrix_blk_grid_rows, std::vector<int>(matrix_blk_grid_cols));

    std::vector<block<T>> submatrix_loc_blocks;
    submatrix_loc_blocks.reserve(matrix_blk_grid_rows *
                                 matrix_blk_grid_cols); // max blocks if 1 proc

    // Iterate over the grid of blocks.
    //
    for (int i = 0; i < matrix_blk_grid_rows; ++i) {
        int rank_row = i % ranks_grid.row;
        for (int j = 0; j < matrix_blk_grid_cols; ++j) {
            int rank_col = j % ranks_grid.col;

            matrix_owners[i][j] = rank_from_grid({rank_row, rank_col},
                                                 ranks_grid,
                                                 ranks_grid_ordering,
                                                 matrix_rank_src);

            // If block belongs to current rank
            //
            if (matrix_owners[i][j] == rank) {
                submatrix_loc_blocks.push_back(
                    {{matrix_rows_split[i], matrix_rows_split[i + 1]}, // rows
                     {matrix_cols_split[j], matrix_cols_split[j + 1]}, // cols
                     {i, j},
                     ptr, // TODO: extract offset
                     lld});
            }
        }
    }

    // ----------- SUBMATRIX -------------------

    // If submatrix was requested
    //
    if (submatrix_begin.row == 1 && submatrix_begin.col == 1) {
        // TODO: return matrix layout
    }

    submatrix_begin.row--;
    submatrix_begin.col--;
    scalapack::elem_grid_coord submatrix_end{
        submatrix_begin.row + submatrix_shape.row,
        submatrix_begin.col + submatrix_shape.col};

    std::vector<int> submatrix_rows_split =
        line_split(submatrix_begin[0], submatrix_end[0], blk_shape[0]);
    std::vector<int> submatrix_cols_split =
        line_split(submatrix_begin[1], submatrix_end[1], blk_shape[1]);

    int submatrix_blk_grid_rows =
        static_cast<int>(submatrix_rows_split.size() - 1);
    int submatrix_blk_grid_cols =
        static_cast<int>(submatrix_cols_split.size() - 1);

    std::vector<std::vector<int>> owners(
        submatrix_blk_grid_rows, std::vector<int>(submatrix_blk_grid_cols));

    // Find out the process to which submatrix_begin belongs.
    //
    scalapack::rank_grid_coord submatrix_rank_src;

    std::vector<block<T>> submatrix_loc_blocks;
    submatrix_loc_blocks.reserve(
        submatrix_blk_grid_cols *
        submatrix_blk_grid_rows); // max blocks if 1 proc

    // Iterate over the grid of blocks.
    //
    for (int i = 0; i < submatrix_blk_grid_rows; ++i) {
        int rank_row = i % ranks_grid.row;
        for (int j = 0; j < submatrix_blk_grid_cols; ++j) {
            int rank_col = j % ranks_grid.col;

            owners[i][j] = rank_from_grid({rank_row, rank_col},
                                          ranks_grid,
                                          ranks_grid_ordering,
                                          submatrix_rank_src);

            // If block belongs to current rank
            //
            if (owners[i][j] == rank) {
                submatrix_loc_blocks.push_back(
                    {{submatrix_rows_split[i],
                      submatrix_rows_split[i + 1]}, // rows
                     {submatrix_cols_split[j],
                      submatrix_cols_split[j + 1]}, // cols
                     {i, j},
                     ptr, // TODO: I need information on the enclosing
                          //       block
                     lld});
            }
        }
    }
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
                   char transpose,
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
                              transpose,
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

    char transpose = 'N';

    return get_scalapack_grid(stride, // local leading dim
                              m_dim,  // global matrix size
                              {1, 1}, // start of submatrix
                              m_dim,  // dim of submatrix
                              b_dim,  // block dimension
                              r_grid,
                              rank_grid_ordering,
                              transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
                   char transpose,
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
