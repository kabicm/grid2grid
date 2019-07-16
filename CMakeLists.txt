cmake_minimum_required(VERSION 3.12)

if(NOT DEFINED PROJECT_NAME)
  set(GRID2GRID_NOT_SUBPROJECT ON)
endif()

project(grid2grid VERSION 0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

add_library(grid2grid STATIC "${grid2grid_SOURCE_DIR}/src/block.cpp"
                             "${grid2grid_SOURCE_DIR}/src/grid2D.cpp"
                             "${grid2grid_SOURCE_DIR}/src/interval.cpp"
                             "${grid2grid_SOURCE_DIR}/src/scalapack_layout.cpp"
                             "${grid2grid_SOURCE_DIR}/src/communication_data.cpp"
                             "${grid2grid_SOURCE_DIR}/src/grid_cover.cpp"
                             "${grid2grid_SOURCE_DIR}/src/transform.cpp")
target_include_directories(grid2grid PUBLIC
  $<BUILD_INTERFACE:${grid2grid_SOURCE_DIR}/src>
  )
target_compile_features(grid2grid PUBLIC cxx_std_11)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(grid2grid OpenMP::OpenMP_CXX MPI::MPI_CXX)

if (GRID2GRID_NOT_SUBPROJECT)
  add_subdirectory(libs)
  include(CTest)
  enable_testing()
  add_subdirectory(examples)
endif()

if (GRID2GRID_NOT_SUBPROJECT)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(TARGETS grid2grid
            EXPORT grid2grid_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(EXPORT grid2grid_targets
            FILE grid2gridTargets.cmake
            NAMESPACE grid2grid::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grid2grid")

    install(DIRECTORY "${grid2grid_SOURCE_DIR}/src/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hpp")

    write_basic_package_version_file(
      "${grid2grid_BINARY_DIR}/grid2gridConfigVersion.cmake"
      VERSION ${grid2grid_VERSION}
      COMPATIBILITY SameMajorVersion)

    install(FILES "${grid2grid_SOURCE_DIR}/cmake/grid2gridConfig.cmake"
                  "${grid2grid_BINARY_DIR}/grid2gridConfigVersion.cmake"
                  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grid2grid")
endif()
