// assumes column-major ordering inside block
namespace grid2grid {
// ****************
//     BLOCK
// ****************
template <typename T>
block<T>::block(const assigned_grid2D& grid, block_coordinates coord, T *ptr, int stride):
    rows_interval(grid.rows_interval(coord.row)),
    cols_interval(grid.cols_interval(coord.col)),
    coordinates(coord), data(ptr), stride(stride) {}

template <typename T>
block<T>::block(const assigned_grid2D& grid, block_coordinates coord, T *ptr) :
    block(grid, coord, ptr, grid.rows_interval(coord.row).length()) {}

template <typename T>
block<T>::block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr, int stride):
    rows_interval(r_inter), cols_interval(c_inter),
    coordinates(coord), data(ptr), stride(stride) {}

template <typename T>
block<T>::block(interval r_inter, interval c_inter, block_coordinates coord, T *ptr) :
    block(r_inter, c_inter, coord, ptr, r_inter.length()) {}

template <typename T>
block<T>::block(block_range& range, block_coordinates coord, T *ptr, int stride) :
    block(range.rows_interval, range.cols_interval, coord, ptr, stride) {}

template <typename T>
block<T>::block(block_range& range, block_coordinates coord, T *ptr) :
    block(range.rows_interval, range.cols_interval, coord, ptr) {}

std::ostream& operator<<(std::ostream &os, const block_range &other) {
    os << "rows:" << other.rows_interval << ", cols:" << other.cols_interval << std::endl;
    return os;
}

template <typename T>
block<T> block<T>::subblock(interval r_range, interval c_range) const {
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

template <typename T>
bool block<T>::non_empty() const {
    bool non_empty_intervals = cols_interval.non_empty() && rows_interval.non_empty();
    assert(!non_empty_intervals || data);
    // std::cout << "data = " << data << std::endl;
    return non_empty_intervals;
}

template <typename T>
// implementing comparator
bool block<T>::operator<(const block &other) const {
    return cols_interval.start < other.cols_interval.start ||
           (cols_interval.start == other.cols_interval.start && rows_interval.start < other.rows_interval.start);
}

template <typename T>
int block<T>::n_rows() const {
    return rows_interval.length();
}

template <typename T>
int block<T>::n_cols() const {
    return cols_interval.length();
}

template <typename T>
std::pair<int, int> block<T>::size() const {
    return {n_rows(), n_cols()};
}

template <typename T>
size_t block<T>::total_size() const {
    return n_rows() * n_cols();
}

template <typename T>
std::ostream& operator<<(std::ostream &os, const block<T> &other) {
    return os << "rows: " << other.rows_interval << "cols: " << other.cols_interval << std::endl;
}

// ****************
//   LOCAL BLOCKS
// ****************
template <typename T>
local_blocks<T>::local_blocks(std::vector<block<T>> &&blocks) :
        blocks(std::forward<std::vector<block<T>>>(blocks)) {
    for (const auto &b : blocks) {
        this->total_size += b.total_size();
    }
};

template <typename T>
block<T>& local_blocks<T>::get_block(int i) {
    return blocks[i];
}
template <typename T>
const block<T>& local_blocks<T>::get_block(int i) const {
    return blocks[i];
}

template <typename T>
int local_blocks<T>::num_blocks() const {
    return blocks.size();
}

template <typename T>
size_t local_blocks<T>::size() const {
    return total_size;
}

template <typename T>
std::ostream& operator<<(std::ostream &os, const local_blocks<T> &other) {
    for (unsigned i = 0; i < (unsigned) other.num_blocks(); ++i) {
        os << "block " << i << ":\n" << other.get_block(i) << std::endl;
    }
    return os;
}
}

