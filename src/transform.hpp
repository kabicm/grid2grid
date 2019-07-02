#pragma once
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <utility>
#include <mpi.h>
#include <cmath>
#include <chrono>
#include <assert.h>

#include "scalapack_layout.hpp"
#include "memory_utils.hpp"
#include "interval.hpp"
#include "grid2D.hpp"
#include "block.hpp"
#include "communication_data.hpp"
#include "grid_layout.hpp"
#include "grid_cover.hpp"
#include "grid_cover.hpp"

namespace grid2grid {
template <typename T>
grid_layout<T> get_scalapack_grid(scalapack::data_layout& layout,
                               T *ptr, int rank);
template<typename T>
grid_layout<T> get_scalapack_grid(int lld_m_dim, // local leading dim
                               scalapack::matrix_dim m_dim, // global matrix size
                               scalapack::elem_grid_coord ij, // start of submatrix
                               scalapack::matrix_dim subm_dim, // dim of submatrix
                               scalapack::block_dim b_dim, // block dimension
                               scalapack::rank_decomposition r_grid,
                               scalapack::ordering rank_grid_ordering,
                               bool transposed,
                               scalapack::rank_grid_coord rank_src,
                               T* ptr, int rank);

template <typename T>
void transform(grid_layout<T>& initial_layout, grid_layout<T>& final_layout, MPI_Comm comm);
}

#include "transform.cpp"
