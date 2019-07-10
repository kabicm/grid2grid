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
target_include_directories(grid2grid PUBLIC "${grid2grid_SOURCE_DIR}/src")
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
