#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
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
        return rows_interval.empty() || cols_interval.empty();
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

inline
std::ostream& operator<<(std::ostream &os, const block_range &other) {
    os << "rows:" << other.rows_interval << ", cols:" << other.cols_interval << std::endl;
    return os;
}

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

    block(const assigned_grid2D& grid, block_coordinates coord, T *ptr, int stride):
        rows_interval(grid.rows_interval(coord.row)),
        cols_interval(grid.cols_interval(coord.col)),
        coordinates(coord), data(ptr), stride(stride) {}

    block(const assigned_grid2D& grid, block_coordinates coord, T *ptr) :
        block(grid, coord, ptr, grid.rows_interval(coord.row).length()) {}

    block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr, int stride):
        rows_interval(r_inter), cols_interval(c_inter),
        coordinates(coord), data(ptr), stride(stride) {}

    block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr) :
        block(r_inter, c_inter, coord, ptr, r_inter.length()) {}

    block(block_range& range, block_coordinates coord, T *ptr, int stride) :
        block(range.rows_interval, range.cols_interval, coord, ptr, stride) {}

    block(block_range& range, block_coordinates coord, T *ptr) :
        block(range.rows_interval, range.cols_interval, coord, ptr) {}

    // finds the index of the interval inter in splits
    int interval_index(const std::vector<int>& splits, interval inter) {
        auto ptr = std::lower_bound(splits.begin(), splits.end(), inter.start);
        int index = std::distance(splits.begin(), ptr);
        return index;
    }

    // without coordinates
    block(const assigned_grid2D& grid,
        interval r_inter, interval c_inter, T *ptr, int stride):
        rows_interval(r_inter), cols_interval(c_inter), data(ptr), stride(stride) {
        // compute the coordinates based on the grid and intervals
        int row_coord = interval_index(grid.grid().rows_split, rows_interval);
        int col_coord = interval_index(grid.grid().cols_split, cols_interval);
        coordinates = block_coordinates(row_coord, col_coord);
    }

    block(const assigned_grid2D& grid,
        interval r_inter, interval c_inter, T *ptr) :
        block(grid, r_inter, c_inter, ptr, r_inter.length()) {}

    block(const assigned_grid2D& grid,
        block_range& range, T *ptr, int stride) :
        block(grid, range.rows_interval, range.cols_interval, ptr, stride) {}

    block(const assigned_grid2D& grid, block_range& range, T *ptr) :
        block(grid, range.rows_interval, range.cols_interval, ptr) {}

    block<T> subblock(interval r_range, interval c_range) const {
        if (!rows_interval.contains(r_range) || !cols_interval.contains(c_range)) {
            std::cout << "BLOCK: row_interval = " << rows_interval << ", column_interval = " << cols_interval
                      << std::endl;
            std::cout << "SUBBLOCK: row_interval = " << r_range << ", column_interval = " << c_range << std::endl;
            throw std::runtime_error("ERROR: current block does not contain requested subblock.");
        }
        // column-major ordering inside block assumed here
        T *ptr = data + (c_range.start - cols_interval.start) * stride
                 + (r_range.start - rows_interval.start);
        // std::cout << "stride = " << stride << std::endl;
        // std::cout << "ptr offset = " << (ptr - data) << std::endl;
        return {r_range, c_range, coordinates, ptr, stride};
    }

    bool non_empty() const {
        bool non_empty_intervals = cols_interval.non_empty() && rows_interval.non_empty();
        assert(!non_empty_intervals || data);
        // std::cout << "data = " << data << std::endl;
        return non_empty_intervals;
    }

    // implementing comparator
    bool operator<(const block &other) const {
        return cols_interval.start < other.cols_interval.start ||
               (cols_interval.start == other.cols_interval.start && rows_interval.start < other.rows_interval.start);
    }

    int n_rows() const {
        return rows_interval.length();
    }

    int n_cols() const {
        return cols_interval.length();
    }

    std::pair<int, int> size() const {
        return {n_rows(), n_cols()};
    }

    size_t total_size() const {
        return n_rows() * n_cols();
    }
};

template <typename T>
inline
std::ostream& operator<<(std::ostream &os, const block<T> &other) {
    return os << "rows: " << other.rows_interval << "cols: " << other.cols_interval << std::endl;
}

template <typename T>
class local_blocks {
public:
    local_blocks() = default;
    local_blocks(std::vector<block<T>> &&blocks) :
        blocks(std::forward<std::vector<block<T>>>(blocks)) {
        for (const auto &b : blocks) {
            this->total_size += b.total_size();
        }
    }

    block<T>& get_block(int i) {
        return blocks[i];
    }

    const block<T>& get_block(int i) const {
        return blocks[i];
    }

    int num_blocks() const {
        return blocks.size();
    }

    size_t size() const {
        return total_size;
    }

private:
    std::vector<block<T>> blocks;
    size_t total_size = 0;
};

template<typename T>
inline
std::ostream& operator<<(std::ostream &os, const local_blocks<T> &other) {
    for (unsigned i = 0; i < (unsigned) other.num_blocks(); ++i) {
        os << "block " << i << ":\n" << other.get_block(i) << std::endl;
    }
    return os;
}
}
