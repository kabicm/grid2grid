###############
#  Src files  #
###############

set(sources
    mpi_type_wrapper.hpp
    cantor_mapping.hpp
    block.hpp
    grid2D.cpp
    interval.cpp
    scalapack_layout.cpp
    communication_data.hpp
    grid_cover.hpp
    transform.hpp)

###################
#     Library     #
###################
add_library(grid2grid ${sources})
target_link_libraries(grid2grid ${DEPENDENCIES})
