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

template <typename T>
void transform(grid_layout<T>& initial_layout, grid_layout<T>& final_layout, MPI_Comm comm);
}

#include "transform.cpp"
