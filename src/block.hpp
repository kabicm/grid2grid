#pragma once
#include <vector>
#include <iostream>
#include "interval.hpp"
#include "grid2D.hpp"

namespace grid2grid {
struct block_coordinates {
    int row = 0;
    int col = 0;
    block_coordinates() = default;
    block_coordinates(int r, int c): row(r), col(c) {}
};

// assumes column-major ordering inside block
template <typename T>
struct block {
    // start and end index of the block
    interval rows_interval;
    interval cols_interval;

    block_coordinates coordinates;

    T *data = nullptr;
    int stride = 0;

    block() = default;

    block(const assigned_grid2D& grid, block_coordinates coord, T *ptr, int stride);
    block(const assigned_grid2D& grid, block_coordinates coord, T *ptr);

    block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr, int stride);
    block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr);

    block subblock(interval r_range, interval c_range) const;

    bool non_empty() const;

    // implementing comparator
    bool operator<(const block &other) const;

    int n_rows() const;

    int n_cols() const;

    std::pair<int, int> size() const;

    size_t total_size() const;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const block<T> &other);

template <typename T>
class local_blocks {
public:
    local_blocks() = default;
    local_blocks(std::vector<block<T>> &&blocks);

    block<T>& get_block(int i);
    const block<T>& get_block(int i) const;

    int num_blocks() const;

    size_t size() const;

private:
    std::vector<block<T>> blocks;
    size_t total_size = 0;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const local_blocks<T> &other);
}

#include "block.cpp"
