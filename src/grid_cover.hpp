#pragma once
#include <vector>
#include "interval.hpp"
#include "grid2D.hpp"
#include "block.hpp"

namespace grid2grid {
    struct interval_cover {
        int start_index = 0;
        int end_index = 0;

        interval_cover() = default;

        interval_cover(int start_idx, int end_idx) :
                start_index(start_idx), end_index(end_idx) {}
    };

    std::ostream& operator<<(std::ostream& os, const interval_cover& other);

    struct block_cover {
        interval_cover rows_cover;
        interval_cover cols_cover;

        block_cover() = default;

        block_cover(interval_cover rows_cover, interval_cover cols_cover) :
                rows_cover(rows_cover), cols_cover(cols_cover) {}
    };

    std::vector <interval_cover> get_decomp_cover(
        const std::vector<int> &decomp_blue,
        const std::vector<int> &decomp_red);

    struct grid_cover {
        std::vector<interval_cover> rows_cover;
        std::vector<interval_cover> cols_cover;

        grid_cover() = default;

        grid_cover(const grid2D& g1, const grid2D& g2);

        template <typename T>
        block_cover decompose_block(const block<T>& b);
    };
}

#include "grid_cover.cpp"

