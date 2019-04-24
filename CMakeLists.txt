cmake_minimum_required(VERSION 3.5)
PROJECT(grid2grid)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")

set(WITH_OPENMP ON CACHE BOOL "Use OpenMP.")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

#################################
#  Generic Compilation options  #
#################################
# Compiler must support c++14
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
else ()
    message(FATAL_ERROR "This library supports only GNU compilers.")
endif ()

############
# MPI Part #
############
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(BEFORE SYSTEM ${MPI_INCLUDE_PATH})
    set(MPI_DEPS ${MPI_CXX_LIBRARIES})
endif ()

###############
# OpenMP Part #
###############
if (WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        set(OPENMP_DEPS OpenMP::OpenMP_CXX)
    else()
        set(OPENMP_DEPS "")
    endif()
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} {OpenMP_EXE_LINKER_FLAGS}")
endif()

set(DEPENDENCIES ${MPI_DEPS} ${OPENMP_DEPS})

#############
#  Testing  #
#############
include(CTest)
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(examples)
