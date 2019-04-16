namespace grid2grid {
std::vector<interval_cover> get_decomp_cover(const std::vector<int> &decomp_blue, const std::vector<int> &decomp_red) {
#ifdef DEBUG
    std::cout << "decomp_blue = " << std::endl;
    for (const auto& v : decomp_blue) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;

    std::cout << "decomp_red = " << std::endl;
    for (const auto& v : decomp_red) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
#endif

    std::vector<interval_cover> cover;
    cover.reserve(decomp_blue.size() - 1);

    int r = decomp_red[0];
    int r_prev = decomp_red[0];
    int i_red = 1;
    int i_prev = 0;

    int b = decomp_blue[0];

    int i_blue = 1;
    while ((size_t) i_blue < decomp_blue.size()) {
        int r_left = r_prev;
        if (i_blue > 1) {
            while (r_left < decomp_blue[i_blue - 1]) {
                i_prev++;
                r_left = decomp_red[i_prev];
            }
            if (r_left > decomp_blue[i_blue - 1]) {
                i_prev--;
                r_left = decomp_red[i_prev];
            }
        }
        int i_left = i_prev;
        // std::cout << "r_left = " << r_left << std::endl;
        b = decomp_blue[i_blue];
        // std::cout << "b = " << b << std::endl;
        r = decomp_red[i_red];
        while (r < b) {
            r_prev = r;
            i_prev = i_red;
            i_red++;
            r = decomp_red[i_red];
        }

        assert(r_left < r);

        cover.push_back({i_left, i_red});
        i_blue++;
    }
#ifdef DEBUG
    std::cout << "cover = " << std::endl;
    for (const auto& v : cover) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
#endif

    return cover;
}

std::ostream& operator<<(std::ostream& os, const interval_cover& other) {
    os << "interval_cover[" << other.start_index << ", " << other.end_index << "]";
    return os;
}

grid_cover::grid_cover(const grid2D& g1, const grid2D& g2) {
    rows_cover = get_decomp_cover(g1.rows_split, g2.rows_split);
    cols_cover = get_decomp_cover(g1.cols_split, g2.cols_split);
}

template <typename T>
block_cover grid_cover::decompose_block(const block<T>& b) {
    int row_index = b.coordinates.row;
    int col_index = b.coordinates.col;
    if (row_index < 0 || (size_t) row_index >= rows_cover.size() ||
        col_index < 0 || (size_t) col_index >= cols_cover.size()) {
        throw std::runtime_error("Error in decompose block. Block coordinates do not belong to the grid cover.");
    }
    return {rows_cover[row_index], cols_cover[col_index]};
}
}
