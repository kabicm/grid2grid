#pragma once

#include <grid2grid/block.hpp>
#include <grid2grid/communication_data.hpp>
#include <grid2grid/grid2D.hpp>
#include <grid2grid/grid_cover.hpp>
#include <grid2grid/grid_layout.hpp>
#include <grid2grid/interval.hpp>
#include <grid2grid/memory_utils.hpp>
#include <grid2grid/scalapack_layout.hpp>

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace grid2grid {
// template <typename T>
// grid_layout<T> get_scalapack_grid(scalapack::data_layout& layout,
//                               T *ptr, int rank);

// The following two definitions are the same with the exception of a const
//
template <typename T>
grid_layout<T>
get_scalapack_grid(int lld_m_dim,                  // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   char transpose,
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
                   char transpose,
                   scalapack::rank_grid_coord rank_src,
                   const T *ptr,
                   const int rank);

// There is not submatrix support here.
//
template <typename T>
grid_layout<T> get_scalapack_grid(scalapack::matrix_dim m_dim,
                                  scalapack::block_dim b_dim,
                                  scalapack::rank_decomposition r_grid,
                                  scalapack::ordering rank_grid_ordering,
                                  T *ptr,
                                  int rank);

// Provides a more conveninet wasy to pass arguments. There is no submatrix
// support.
//
template <typename T>
grid_layout<T>
get_scalapack_grid(scalapack::data_layout &layout, T *ptr, int rank);

template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm);

} // namespace grid2grid
