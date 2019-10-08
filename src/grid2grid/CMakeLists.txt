add_library(grid2grid STATIC block.cpp
                             grid2D.cpp
                             interval.cpp
                             scalapack_layout.cpp
                             communication_data.cpp
                             grid_cover.cpp
                             ranks_reordering.hpp
                             transform.cpp
                             transformer.hpp
                             )

target_include_directories(grid2grid PUBLIC
  $<BUILD_INTERFACE:${grid2grid_SOURCE_DIR}/src>
  )

find_package(blaze)
if( blaze_FOUND )
    message("BLAZE FOUND")
    # add_library( blaze_target INTERFACE )
    target_link_libraries( grid2grid PUBLIC blaze::blaze )
endif()

target_link_libraries(grid2grid PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX blaze::blaze)

if(GRID2GRID_WITH_PROFILING)
    target_link_libraries(grid2grid PRIVATE 
        $<BUILD_INTERFACE:semiprof>
    )
    target_compile_definitions(grid2grid PRIVATE 
        $<BUILD_INTERFACE:GRID2GRID_WITH_PROFILING>
    )
endif()

target_compile_features(grid2grid PUBLIC cxx_std_11)

if(GRID2GRID_WITH_INSTALL)
    install(TARGETS grid2grid
            EXPORT grid2grid_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
