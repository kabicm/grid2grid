cmake_minimum_required(VERSION 3.12)

include(GNUInstallDirs)

set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
    set(MASTER_PROJECT ON)
endif()

option(GRID2GRID_WITH_EXAMPLES "Enable examples." ${MASTER_PROJECT})
option(GRID2GRID_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
option(GRID2GRID_WITH_PROFILING "Enable profiling." OFF)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

project(grid2grid VERSION 0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

# Dependencies
#
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Grid2grid
#
add_subdirectory(src/grid2grid)

if(GRID2GRID_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${GRID2GRID_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif()

if (GRID2GRID_WITH_EXAMPLES)
    add_subdirectory(libs/options)
    enable_testing()
    add_subdirectory(examples)
endif()

if(GRID2GRID_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    
    install(EXPORT grid2grid_targets
            FILE grid2gridTargets.cmake
            NAMESPACE grid2grid::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grid2grid")
    
    install(DIRECTORY "${grid2grid_SOURCE_DIR}/src/grid2grid"
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

