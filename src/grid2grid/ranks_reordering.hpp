#pragma once

#include <grid2grid/comm_volume.hpp>
#include <unordered_set>
#include <vector>
#include <limits>
#include <cassert>
#include <algorithm>

namespace grid2grid {
std::vector<int> optimal_reordering(comm_volume& comm_volume, int n_ranks);
}


