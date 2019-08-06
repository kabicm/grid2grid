add_library(grid2grid STATIC block.cpp
                             grid2D.cpp
                             interval.cpp
                             scalapack_layout.cpp
                             communication_data.cpp
                             grid_cover.cpp
                             transform.cpp)

target_include_directories(grid2grid PUBLIC
  $<BUILD_INTERFACE:${grid2grid_SOURCE_DIR}/src>
  )

target_link_libraries(grid2grid OpenMP::OpenMP_CXX MPI::MPI_CXX)

target_compile_features(grid2grid PUBLIC cxx_std_11)

if(GRID2GRID_WITH_INSTALL)
    install(TARGETS grid2grid
            EXPORT grid2grid_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
