#pragma once

#include "block.hpp"
#include "memory_utils.hpp"
#include <chrono>
#include <memory>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

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
    // std::vector<double, cosma::mpi_allocator<double>> buffer;
    std::vector<int> dspls;
    std::vector<int> counts;
    std::vector<message<T>> messages;
    int n_ranks = 0;
    int total_size = 0;

    communication_data() = default;

    communication_data(std::vector<message<T>> &&msgs, int n_ranks);

    void copy_to_buffer();

    void copy_from_buffer();

    T *data();

  private:
    std::vector<int> offset_per_message;
};
} // namespace grid2grid

#include "communication_data.cpp"
