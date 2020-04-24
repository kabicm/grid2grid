#pragma once
#include <grid2grid/block.hpp>
#include <grid2grid/memory_utils.hpp>

#include <chrono>
#include <memory>
#include <vector>

namespace grid2grid {
template <typename T>
class message {

  public:
    message() = default;

    message(block<T> b, int rank);

    block<T> get_block() const;

    int get_rank() const;

    // implementing comparator
    bool operator<(const message<T> &other) const;

  private:
    block<T> b;
    int rank = 0;
};

template <typename T>
class communication_data {
  public:
    std::unique_ptr<T[]> buffer;
    T* assigned_buffer = nullptr;
    // std::vector<double, cosma::mpi_allocator<double>> buffer;
    std::vector<int> dspls;
    std::vector<int> counts;
    // mpi_messages are the ones that have to be
    // communicated to a different rank
    std::vector<message<T>> mpi_messages;
    // blocks which should be copied locally,
    // and not through MPI
    std::vector<block<T>> local_blocks;
    int n_ranks = 0;
    int total_size = 0;
    int my_rank;
    int n_packed_messages = 0;

    communication_data() = default;

    communication_data(std::vector<message<T>> &msgs, int my_rank, int n_ranks);

    // copy all mpi_messages to buffer
    void copy_to_buffer();
    // copy mpi_messages within the idx-th package
    // a package includes all mpi_messages
    // to be sent to the same rank
    void copy_to_buffer(int idx);

    // copy all mpi_messages from buffer
    void copy_from_buffer();
    // copy mpi_messages within the idx-th package
    // a package includes all mpi_messages
    // received from the same rank
    void copy_from_buffer(int idx);

    T *data() const;

    // check if the user provided the mpi buffer
    // and if not: allocates a new one
    void initialize_buffer_ptr();

    void partition_messages();

    // user-provided buffer to be used
    void assign_workspace(T* ptr);

  private:
    std::vector<int> package_ticks;
    std::vector<int> offset_per_message;
};

template <typename T>
void copy_local_blocks(std::vector<block<T>>& from, std::vector<block<T>>& to);
} // namespace grid2grid
