#pragma once

#include <grid2grid/comm_volume.hpp>
#include <vector>

namespace grid2grid {
std::vector<int> optimal_reordering(comm_volume& comm_volume, int n_ranks, bool& reordered);
}


