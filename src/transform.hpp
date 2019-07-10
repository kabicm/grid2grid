#pragma once
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "block.hpp"
#include "communication_data.hpp"
#include "grid2D.hpp"
#include "grid_cover.hpp"
#include "grid_layout.hpp"
#include "interval.hpp"
#include "memory_utils.hpp"
#include "scalapack_layout.hpp"

namespace grid2grid {
// template <typename T>
// grid_layout<T> get_scalapack_grid(scalapack::data_layout& layout,
//                               T *ptr, int rank);

template <typename T>
grid_layout<T>
get_scalapack_grid(int lld_m_dim,                  // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   T *ptr,
                   const int rank);

template <typename T>
grid_layout<T>
get_scalapack_grid(int lld_m_dim,                  // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   bool transposed,
                   scalapack::rank_grid_coord rank_src,
                   const T *ptr,
                   const int rank);

template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm);

} // namespace grid2grid

#include "transform.cpp"
