#include <grid2grid/transform.hpp>
#include <grid2grid/profiler.hpp>
#include <complex>
#include <thread>
#include <unordered_map>

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

std::unordered_map<int, int> rank_to_comm_vol_for_block(const assigned_grid2D& g_init,
                                                        const block_coordinates &b_coord,
                                                        grid_cover &g_cover,
                                                        const assigned_grid2D& g_final) {
    // std::cout << "decomposing block " << b << std::endl;
    block_cover b_cover = g_cover.decompose_block(b_coord);

    int row_first = b_cover.rows_cover.start_index;
    int row_last = b_cover.rows_cover.end_index;

    int col_first = b_cover.cols_cover.start_index;
    int col_last = b_cover.cols_cover.end_index;

    auto rows_interval = g_init.rows_interval(b_coord.row);
    auto cols_interval = g_init.cols_interval(b_coord.col);

    std::unordered_map<int, int> comm_vol;

    int row_start = rows_interval.start;
    // use start of the interval to get the rank and the end of the interval
    // to get the block which has to be sent
    // skip the last element
    for (int i = row_first; i < row_last; ++i) {
        int row_end = std::min(g_final.grid().rows_split[i + 1], rows_interval.end);

        int col_start = cols_interval.start;
        for (int j = col_first; j < col_last; ++j) {
            // use i, j to find out the rank
            int rank = g_final.owner(i, j);
            // std::cout << "owner of block " << i << ", " << j << " is " <<
            // rank << std::endl;

            // use i+1 and j+1 to find out the block
            int col_end =
                std::min(g_final.grid().cols_split[j + 1], cols_interval.end);

            int size = (row_end - row_start) * (col_end - col_start);
            // if non empty, add this block
            if (size) {
                comm_vol[rank] += size;
            }

            col_start = col_end;
        }
        row_start = row_end;
    }
    return comm_vol;
}

template <typename T>
void merge_messages(std::vector<message<T>> &messages) {
    std::sort(messages.begin(), messages.end());
}

template <typename T>
std::vector<message<T>> decompose_blocks(const grid_layout<T> &init_layout,
                                         const grid_layout<T> &final_layout,
                                         int tag = 0) {
    PE(transform_decompose);
    grid_cover g_overlap(init_layout.grid.grid(), final_layout.grid.grid());

    std::vector<message<T>> messages;

    for (int i = 0; i < init_layout.blocks.num_blocks(); ++i) {
        // std::cout << "decomposing block " << i << " out of " <<
        // init_layout.blocks.num_blocks() << std::endl;
        auto blk = init_layout.blocks.get_block(i);
        blk.tag = tag;
        assert(blk.non_empty());
        std::vector<message<T>> decomposed =
            decompose_block(blk, g_overlap, final_layout.grid);
        messages.insert(messages.end(), decomposed.begin(), decomposed.end());
    }
    PL();
    return messages;
}

template <typename T>
communication_data<T> prepare_to_send(const grid_layout<T> &init_layout,
                                      const grid_layout<T> &final_layout,
                                      int rank) {
    // in case ranks were reordered to minimize the communication
    // this might not be the identity function
    // if (rank == 0) {
    //     std::cout << "prepare to send: changing rank to " << init_layout.reordered_rank(rank) << std::endl;
    // }
    // rank = init_layout.reordered_rank(rank);
    std::vector<message<T>> messages =
        decompose_blocks(init_layout, final_layout);
    merge_messages(messages);
    return communication_data<T>(messages, rank, final_layout.num_ranks());
}

template <typename T>
communication_data<T> prepare_to_send(
                                      std::vector<layout_ref<T>>& from,
                                      std::vector<layout_ref<T>>& to,
                                      int rank) {
    std::vector<message<T>> messages;
    int n_ranks = 0;

    for (unsigned i = 0u; i < from.size(); ++i) {
        auto& init_layout = from[i].get();
        auto& final_layout = to[i].get();
        auto decomposed_blocks = decompose_blocks(init_layout, final_layout, i);
        messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
        n_ranks = std::max(n_ranks, final_layout.num_ranks());
    }
    merge_messages(messages);
    return communication_data<T>(messages, rank, n_ranks);
}

template <typename T>
communication_data<T> prepare_to_recv(const grid_layout<T> &final_layout,
                                      const grid_layout<T> &init_layout,
                                      int rank) {
    std::vector<message<T>> messages =
        decompose_blocks(final_layout, init_layout);
    merge_messages(messages);
    return communication_data<T>(messages, rank, init_layout.num_ranks());
}

template <typename T>
communication_data<T> prepare_to_recv(
                                      std::vector<layout_ref<T>>& to,
                                      std::vector<layout_ref<T>>& from,
                                      int rank) {
    std::vector<message<T>> messages;
    int n_ranks = 0;

    for (unsigned i = 0u; i < from.size(); ++i) {
        auto& init_layout = from[i].get();
        auto& final_layout = to[i].get();
        auto decomposed_blocks = decompose_blocks(final_layout, init_layout, i);
        messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
        n_ranks = std::max(n_ranks, init_layout.num_ranks());
    }
    merge_messages(messages);
    return communication_data<T>(messages, rank, n_ranks);
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
    scalapack::rank_grid_coord ranks_grid_src_coord,
    T *ptr,
    const int rank) {

    assert(submatrix_begin.row >= 1);
    assert(submatrix_begin.col >= 1);

    submatrix_begin.row--;
    submatrix_begin.col--;

    std::vector<int> rows_split =
        line_split(submatrix_begin.row,
                   submatrix_begin.row + submatrix_shape.row,
                   blk_shape.row);
    std::vector<int> cols_split =
        line_split(submatrix_begin.col,
                   submatrix_begin.col + submatrix_shape.col,
                   blk_shape.col);

    int blk_grid_rows = static_cast<int>(rows_split.size() - 1);
    int blk_grid_cols = static_cast<int>(cols_split.size() - 1);

    std::vector<std::vector<int>> owners(blk_grid_rows,
                                         std::vector<int>(blk_grid_cols));
    std::vector<block<T>> loc_blocks;
    loc_blocks.reserve(blk_grid_rows * blk_grid_cols);

    // The begin block grid coordinates of the matrix block which is inside or
    // is split by the submatrix.
    //
    int border_blk_row_begin = submatrix_begin.row / blk_shape.row;
    int border_blk_col_begin = submatrix_begin.col / blk_shape.col;

    scalapack::rank_grid_coord submatrix_rank_grid_src_coord{
        (border_blk_row_begin % ranks_grid.row + ranks_grid_src_coord.row) %
            ranks_grid.row,
        (border_blk_col_begin % ranks_grid.col + ranks_grid_src_coord.col) %
            ranks_grid.col};

    // Iterate over the grid of blocks of the submatrix.
    //
    for (int j = 0; j < blk_grid_cols; ++j) {
        int rank_col =
            (j % ranks_grid.col + submatrix_rank_grid_src_coord.col) %
            ranks_grid.col;
        for (int i = 0; i < blk_grid_rows; ++i) {
            int rank_row =
                (i % ranks_grid.row + submatrix_rank_grid_src_coord.row) %
                ranks_grid.row;

            // The rank to which the block belongs
            //
            owners[i][j] = rank_from_grid(
                {rank_row, rank_col}, ranks_grid, ranks_grid_ordering);

            // If block belongs to current rank
            //
            if (owners[i][j] == rank) {
                // Coordinates of the (border) block within this rank.
                //
                int blk_loc_row = (border_blk_row_begin + i) / ranks_grid.row;
                int blk_loc_col = (border_blk_col_begin + j) / ranks_grid.col;

                // The begin coordinates of the sub-block within the local block
                // in this process.
                //
                int subblk_loc_row = submatrix_begin.row + rows_split[i] -
                                     (border_blk_row_begin + i) * blk_shape.row;
                int subblk_loc_col = submatrix_begin.col + cols_split[j] -
                                     (border_blk_col_begin + j) * blk_shape.col;

                int data_offset =
                    blk_loc_row * blk_shape.row + subblk_loc_row +
                    lld * (blk_loc_col * blk_shape.col + subblk_loc_col);

                loc_blocks.push_back(
                    {{rows_split[i], rows_split[i + 1]}, // rows
                     {cols_split[j], cols_split[j + 1]}, // cols
                     {i, j},                             // blk coords
                     ptr + data_offset,
                     lld});
            }
        }
    }

    grid2D grid(std::move(rows_split), std::move(cols_split));
    assigned_grid2D assigned_grid(
        std::move(grid), std::move(owners), ranks_grid.n_total());
    local_blocks<T> local_memory(std::move(loc_blocks));
    grid_layout<T> layout(std::move(assigned_grid), std::move(local_memory));
    layout.transpose_or_conjugate(transpose_flag);
    return layout;
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
void exchange(communication_data<T>& send_data, communication_data<T>& recv_data, MPI_Comm comm) {
    MPI_Request recv_reqs[recv_data.n_packed_messages];

    PE(transform_irecv);
    int request_idx = 0;
    // initiate all receives
    for (unsigned i = 0u; i < recv_data.n_ranks; ++i) {
        if (recv_data.counts[i]) {
            MPI_Irecv(recv_data.data() + recv_data.dspls[i],
                      recv_data.counts[i],
                      mpi_type_wrapper<T>::type(),
                      i, 0, comm,
                      &recv_reqs[request_idx]);
            ++request_idx;
        }
    }
    PL();

    PE(transform_packing)
    // copy blocks to temporary send buffers
    send_data.copy_to_buffer();
    PL();

    PE(transform_isend);
    request_idx = 0;
    MPI_Request send_reqs[send_data.n_packed_messages];
    // initiate all sends
    for (unsigned i = 0u; i < send_data.n_ranks; ++i) {
        if (send_data.counts[i]) {
            MPI_Isend(send_data.data() + send_data.dspls[i], 
                      send_data.counts[i],
                      mpi_type_wrapper<T>::type(),
                      i, 0, comm,
                      &send_reqs[request_idx]);
            ++request_idx;
        }
    }
    PL();

    PE(transform_localblocks);
    // copy local data (that are on the same rank in both initial and final layout)
    // this is independent of MPI and can be executed in parallel
    copy_local_blocks(send_data.local_blocks, recv_data.local_blocks);
    PL();

    PE(transform_waitallirecv);
    MPI_Waitall(recv_data.n_packed_messages, recv_reqs, MPI_STATUSES_IGNORE);
    PL();

    PE(transform_unpacking);
    // copy the received data back to the final layout
    recv_data.copy_from_buffer();
    PL();

    PE(transform_waitallisend);
    // finish up the pending send requests
    MPI_Waitall(send_data.n_packed_messages, send_reqs, MPI_STATUSES_IGNORE);
    PL();
}

template <typename T>
void exchange_async(communication_data<T>& send_data, communication_data<T>& recv_data, MPI_Comm comm) {
    PE(transform_irecv);
    MPI_Request recv_reqs[recv_data.n_packed_messages];
    int request_idx = 0;
    // initiate all receives
    for (unsigned i = 0u; i < recv_data.n_ranks; ++i) {
        if (recv_data.counts[i]) {
            MPI_Irecv(recv_data.data() + recv_data.dspls[i],
                      recv_data.counts[i],
                      mpi_type_wrapper<T>::type(),
                      i, 0, comm,
                      &recv_reqs[request_idx]);
            ++request_idx;
        }
    }
    PL();

    PE(transform_packing);
    // copy blocks to temporary send buffers
    send_data.copy_to_buffer();
    PL();

    PE(transform_isend);
    MPI_Request send_reqs[send_data.n_packed_messages];
    request_idx = 0;
    // initiate all sends
    for (unsigned i = 0u; i < send_data.n_ranks; ++i) {
        if (send_data.counts[i]) {
            MPI_Isend(send_data.data() + send_data.dspls[i], 
                      send_data.counts[i],
                      mpi_type_wrapper<T>::type(),
                      i, 0, comm,
                      &send_reqs[request_idx]);
            ++request_idx;
        }
    }
    PL();

    PE(transform_localblocks);
    // copy local data (that are on the same rank in both initial and final layout)
    // this is independent of MPI and can be executed in parallel
    copy_local_blocks(send_data.local_blocks, recv_data.local_blocks);
    PL();


    // wait for any package and immediately unpack it
    for (unsigned i = 0u; i < recv_data.n_packed_messages; ++i) {
        int idx;
        PE(transform_waitany);
        MPI_Waitany(recv_data.n_packed_messages,
                    recv_reqs,
                    &idx,
                    MPI_STATUS_IGNORE);
        PL();
        PE(transform_unpacking);
        // unpack the package that arrived
        recv_data.copy_from_buffer(idx);
        PL();
    }

    PE(transform_waitall);
    // finish up the send requests since all the receive requests are finished
    MPI_Waitall(send_data.n_packed_messages, send_reqs, MPI_STATUSES_IGNORE);
    PL();
}

// std::vector<int> reorder_ranks(grid_layout<T>& initial_layout,
//                                grid_layout<T>& final_layout) {
// 
// 
// }

comm_volume communication_volume(assigned_grid2D& g_init,
                                 assigned_grid2D& g_final) {
    grid_cover g_cover(g_init.grid(), g_final.grid());

    int n_blocks_row = g_init.grid().n_rows;
    int n_blocks_col = g_init.grid().n_cols;

    std::unordered_map<edge_t, int> weights;

    for (int i = 0; i < n_blocks_row; ++i) {
        for (int j = 0; j < n_blocks_col; ++j) {
            auto rank_to_comm_vol = rank_to_comm_vol_for_block(
                g_init, block_coordinates{i, j}, g_cover, g_final);
            int rank = g_init.owner(i, j);

            for (const auto& comm_vol : rank_to_comm_vol) {
                int target_rank = comm_vol.first;
                int weight = comm_vol.second;

                int smaller_rank = std::min(rank, target_rank);
                int larger_rank = std::max(rank, target_rank);

                edge_t edge_between_ranks{smaller_rank, larger_rank};
                // edge_t edge_between_ranks{rank, target_rank};

                weights[edge_between_ranks] += weight;
            }
        }
    }

    return comm_volume(std::move(weights));
}

template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // std::cout << "rank = " << rank << " preparing sending data" << std::endl;
    communication_data<T> send_data =
        prepare_to_send(initial_layout, final_layout, rank);

    // std::cout << "rank = " << rank << " preparing receiving data" << std::endl;
    communication_data<T> recv_data =
        prepare_to_recv(final_layout, initial_layout, rank);

    // auto comm_volume = communication_volume(initial_layout, final_layout);

    // if (rank == 0) {
    //     for (const auto& edge : comm_volume) {
    //         std::cout << edge.src() << "->" << edge.dest() << " = " << edge.weight() << std::endl;
    //     }
    // }

#ifdef DEBUG
    if (rank == 0) {
        std::cout << "send buffer content: " << std::endl;
        for (int i = 0; i < send_data.total_size; ++i) {
            // std::pair<int, int> el =
            // math_utils::invert_cantor_pairing((int)send_data.buffer[i]);
            std::cout << send_data.buffer[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif

    // perform the communication
    exchange_async(send_data, recv_data, comm);
#ifdef DEBUG
    if (rank == 0) {
        std::cout << "recv buffer content: " << std::endl;
        for (int i = 0; i < recv_data.total_size; ++i) {
            // std::pair<int, int> el =
            // math_utils::invert_cantor_pairing((int)recv_data.buffer[i]);
            std::cout << recv_data.buffer[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif
}

template <typename T>
void transform(std::vector<layout_ref<T>>& from,
               std::vector<layout_ref<T>>& to,
               MPI_Comm comm) {
    assert(from.size() == to.size());

    int rank;
    MPI_Comm_rank(comm, &rank);

    auto send_data = prepare_to_send(from, to, rank);
    auto recv_data = prepare_to_recv(to, from, rank);

    exchange_async(send_data, recv_data, comm);
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

// explicit instantiation of transform with vectors
template void transform<float>(std::vector<layout_ref<float>>& initial_layouts,
                               std::vector<layout_ref<float>>& final_layouts,
                               MPI_Comm comm);

template void transform<double>(std::vector<layout_ref<double>>& initial_layouts,
                                std::vector<layout_ref<double>>& final_layouts,
                                MPI_Comm comm);

template void transform<std::complex<float>>(
                               std::vector<layout_ref<std::complex<float>>>& initial_layouts,
                               std::vector<layout_ref<std::complex<float>>>& final_layouts,
                               MPI_Comm comm);

template void transform<std::complex<double>>(
                               std::vector<layout_ref<std::complex<double>>>& initial_layouts,
                               std::vector<layout_ref<std::complex<double>>>& final_layouts,
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
