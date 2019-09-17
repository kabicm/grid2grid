#pragma once
#include <grid2grid/block.hpp>

#include <algorithm>
#include <cstring>
#include <complex>
#include <cmath>
#include <type_traits>
#include <utility>

namespace grid2grid {
namespace memory {

// copies n entries of elem_type from src_ptr to desc_ptr
template <typename elem_type>
void copy(std::size_t n, const elem_type *src_ptr, elem_type *dest_ptr) {
    static_assert(std::is_trivially_copyable<elem_type>(),
                  "Element type must be trivially copyable!");
    std::memcpy(dest_ptr, src_ptr, sizeof(elem_type) * n);
}

// copies 2D block of given size from src_ptr with stride ld_src
// to dest_ptr with stride ld_dest
template <class elem_type>
void copy2D(const std::pair<size_t, size_t> &block_dim,
            const elem_type *src_ptr,
            int ld_src,
            elem_type *dest_ptr,
            int ld_dest,
            bool col_major = true) {
    static_assert(std::is_trivially_copyable<elem_type>(),
                  "Element type must be trivially copyable!");
    auto block_size = block_dim.first * block_dim.second;
    // std::cout << "invoking copy2D." << std::endl;
    if (!block_size) {
        return;
    }

    auto dim = block_dim;
    if (!col_major) {
        dim = std::make_pair(block_dim.second, block_dim.first);
    }

    // if not strided, copy in a single piece
    if (dim.first == (size_t)ld_src &&
        dim.first == (size_t)ld_dest) {
        copy(block_size, src_ptr, dest_ptr);
    } else {
        // if strided, copy column-by-column
        for (size_t col = 0; col < dim.second; ++col) {
            copy(dim.first,
                 src_ptr + ld_src * col,
                 dest_ptr + ld_dest * col);
        }
    }
}


// copy from block to MPI send buffer
template <typename T>
void copy_and_transpose(const block<T> b, T* dest_ptr, int dest_stride) {
    static_assert(std::is_trivially_copyable<T>(),
            "Element type must be trivially copyable!");
    assert(b.non_empty());
    // n_rows and n_cols before transposing
    int n_rows = b.n_cols();
    int n_cols = b.n_rows();

    int block_dim = std::max(8, 128/(int)sizeof(T));

    std::vector<T> b_elems(block_dim);
    for (int block_i = 0; block_i < n_rows; block_i += block_dim) {
        for (int block_j = 0; block_j < n_cols; block_j += block_dim) {
            if (block_i == block_j) {
                int upper_i = std::min(n_rows, block_i + block_dim);
                int upper_j = std::min(n_cols, block_j + block_dim);
                // #pragma omp task firstprivate(block_i, block_j, dest_ptr, ptr, stride, conj, n_rows_t)
                for (int i = block_i; i < upper_i; ++i) {
                    for (int j = block_j; j < upper_j; ++j) {
                        // (i, j) in the original block, column-major
                        auto el = b.data[j * b.stride + i];
                        // auto el = b.local_element(i, j);
                        // (j, i) in the send buffer, column-major
                        if (b.conjugate_on_copy)
                            el = conjugate(el);
                        b_elems[j-block_j] = el;
                    }
                    for (int j = block_j; j < upper_j; ++j) {
                        dest_ptr[i*dest_stride + j] = b_elems[j-block_j];
                    }
                }
            } else {
                int upper_i = std::min(n_rows, block_i + block_dim);
                int upper_j = std::min(n_cols, block_j + block_dim);
                // #pragma omp task firstprivate(block_i, block_j, dest_ptr, ptr, stride, conj, n_rows_t)
                for (int i = block_i; i < upper_i; ++i) {
                    for (int j = block_j; j < upper_j; ++j) {
                        // (i, j) in the original block, column-major
                        auto el = b.data[j * b.stride + i];
                        // auto el = b.local_element(i, j);
                        // (j, i) in the send buffer, column-major
                        if (b.conjugate_on_copy)
                            el = conjugate(el);
                        dest_ptr[i*dest_stride + j] = b_elems[j-block_j];
                    }
                }
            }
        }
    }
}

template <typename T>
void copy_transposed_back(const T* src_ptr, block<T> b) {
    static_assert(std::is_trivially_copyable<T>(),
                  "Element type must be trivially copyable!");
    for (int i = 0; i < b.n_rows(); ++i) {
        for (int j = 0; j < b.n_cols(); ++j) {
            // (i, j) in the recv buffer, column-major
            // (i, j) in the original block, column-major
            // b.local_element(j, i) = src_ptr[j * b.n_rows() + i];
            b.data[j * b.stride + i] = src_ptr[j * b.n_rows() + i];
        }
    }
}
} // namespace memory
} // namespace grid2grid
