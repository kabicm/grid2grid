#pragma once

#include <grid2grid/communication_data.hpp>
#include <grid2grid/grid2D.hpp>
#include <grid2grid/grid_layout.hpp>
#include <grid2grid/comm_volume.hpp>

#include <mpi.h>

namespace grid2grid {
template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm);

template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               T alpha, T beta,
               MPI_Comm comm);

template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
               std::vector<layout_ref<T>>& final_layouts,
               MPI_Comm comm);

template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
               std::vector<layout_ref<T>>& final_layouts,
               T* alpha, T* beta,
               MPI_Comm comm);

comm_volume communication_volume(assigned_grid2D& g_init,
                                 assigned_grid2D& g_final);

} // namespace grid2grid
