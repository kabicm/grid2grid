#pragma once
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

namespace grid2grid {
namespace memory {

// copies n entries of elem_type from src_ptr to desc_ptr
template <typename elem_type>
void copy(std::size_t n, const elem_type* src_ptr, elem_type* dest_ptr) {
    static_assert(std::is_trivially_copyable<elem_type>(), "Element type must be trivially copyable!");
    std::memcpy(dest_ptr, src_ptr, sizeof(elem_type) * n);
}

// copies 2D block of given size from src_ptr with stride ld_src 
// to dest_ptr with stride ld_dest
template <class elem_type>
void copy2D(const std::pair<size_t, size_t>& block_dim,
        const elem_type* src_ptr, int ld_src,
        elem_type* dest_ptr, int ld_dest) {
    static_assert(std::is_trivially_copyable<elem_type>(), "Element type must be trivially copyable!");
    auto block_size = block_dim.first * block_dim.second;
    // std::cout << "invoking copy2D." << std::endl;
    if (!block_size) {
        return;
    }

    // if not strided, copy in a single piece
    if (block_dim.first == (size_t) ld_src && block_dim.first == (size_t) ld_dest) {
        copy(block_size, src_ptr, dest_ptr);
    } else {
        // if strided, copy column-by-column
        for (unsigned col = 0; col < block_dim.second; ++col) {
            copy(block_dim.first, src_ptr + ld_src * col, dest_ptr + ld_dest * col);
        }
    }
}
}}