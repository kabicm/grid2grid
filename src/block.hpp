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

struct block_range {
    interval rows_interval;
    interval cols_interval;

    block_range() = default;
    block_range(interval r, interval c): rows_interval(r), cols_interval(c) {}

    bool outside_of(const block_range& range) const {
        return (rows_interval.end < range.rows_interval.start
            || rows_interval.start > range.rows_interval.end)
            && (cols_interval.end < range.cols_interval.start
            || cols_interval.end < range.cols_interval.start);
    }

    bool inside(const block_range& range) const {
        return range.rows_interval.start < rows_interval.start
               && range.rows_interval.end > rows_interval.end
               && range.cols_interval.start < cols_interval.start
               && range.cols_interval.end > cols_interval.end;
    }

    bool intersects(const block_range& range) const {
        return !outside_of(range) && !inside(range);
    }

    block_range intersection(const block_range& other) const {
        interval rows_intersection = rows_interval.intersection(other.rows_interval);
        interval cols_intersection = cols_interval.intersection(other.cols_interval);
        return {rows_intersection, cols_intersection};
    }

    bool non_empty() const {
        return rows_interval.non_empty() && cols_interval.non_empty();
    }

    bool empty() const {
        return !non_empty();
    }

    bool operator==(const block_range& other) const {
        if (empty()) {
            return other.empty();
        }
        return rows_interval == other.rows_interval && cols_interval == other.cols_interval;
    }

    bool operator!=(const block_range& other) const {
        return !(*this == other);
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const block_range &other);

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

    block(block_range& range, block_coordinates coord, T *ptr, int stride);
    block(block_range& range, block_coordinates coord, T *ptr);

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
