#pragma once
#include <vector>
#include <cassert>
#include <mpi.h>
#include <grid2grid/transform.hpp>
#include <grid2grid/memory_pool.hpp>
#include <grid2grid/communication_data.hpp>
#include <memory>

namespace grid2grid {
template <typename T>
struct transformer {
    std::vector<layout_ref<T>> from;
    std::vector<layout_ref<T>> to;
    MPI_Comm comm;
    int P;
    int rank;

    communication_data<T> send_data;
    communication_data<T> recv_data;

    // constructor
    transformer() = default;

    // constructor
    transformer(MPI_Comm comm) : comm(comm) {
        MPI_Comm_size(comm, &P);
        MPI_Comm_rank(comm, &rank);
    }

    void schedule(grid_layout<T>& from_layout, grid_layout<T>& to_layout) {
        from.push_back(from_layout);
        to.push_back(to_layout);
    }

    void transform() {
        grid2grid::transform<T>(from, to, comm);
        clear();
    }

    void clear() {
        from.clear();
        to.clear();
    }

    /* 
     * if a user wants to provide a workspace for temporary
     * MPI buffers for packing/unpacking the data
     * then compute_workspace_size() should first be invoked
     * which will return the size of the workspace,
     * and then transform_with_given_workspace should be invoked.
     * The pointer passed to transform_with_given_workspace
     * should be at least the size computed by compute_workspace_size()
     */

    std::size_t compute_workspace_size() {
        send_data = prepare_to_send<T>(from, to, rank);
        recv_data = prepare_to_recv<T>(to, from, rank);
        auto workspace = send_data.total_size + recv_data.total_size;
        return workspace;
    }

    void transform_with_given_workspace(T* ptr) {
        send_data.assign_workspace(ptr);
        recv_data.assign_workspace(ptr + send_data.total_size);

        grid2grid::transform<T>(send_data, recv_data, comm);

        clear();
    }

    void transform_with_memory_pool() {
        auto& pool = get_global_pool<T>();
        pool.resize(compute_workspace_size());

        send_data.assign_workspace(pool.data());
        recv_data.assign_workspace(pool.data() + send_data.total_size);

        grid2grid::transform<T>(send_data, recv_data, comm);

        clear();
        pool.clear();
    }
};
}
