if(NOT TARGET grid2grid::grid2grid)
  include(CMakeFindDependencyMacro)
  find_dependency(MPI)
  find_dependency(OpenMP)

  include("${CMAKE_CURRENT_LIST_DIR}/grid2gridTargets.cmake")
endif()
