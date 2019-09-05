################
#  Build test  #
################
add_executable(scalapack2scalapack scalapack2scalapack.cpp)
target_link_libraries(scalapack2scalapack PRIVATE grid2grid options)

add_executable(scalapack_submatrix scalapack_submatrix.cpp)
target_link_libraries(scalapack_submatrix PRIVATE grid2grid)

add_executable(parallel_scalapack_submatrix parallel_scalapack_submatrix.cpp)
target_link_libraries(parallel_scalapack_submatrix PRIVATE grid2grid)

add_executable(transform_test transform_test.cpp)
target_link_libraries(transform_test PRIVATE grid2grid)

################
#  unit gtests #
################
if (MPI_FOUND)
    add_test(NAME square-blocks-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 ./scalapack2scalapack -m 10 -n 10 -ibm 2 -ibn 2 -fbm 5 -fbn 5 -pm 2 -pn 2)

    add_test(NAME square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 ./scalapack2scalapack -m 10 -n 10 -ibm 2 -ibn 3 -fbm 3 -fbn 5 -pm 2 -pn 2)

    add_test(NAME non_square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 ./scalapack2scalapack -m 10 -n 12 -ibm 2 -ibn 3 -fbm 2 -fbn 6 -pm 2 -pn 2)

    add_test(NAME non-square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 ./scalapack2scalapack -m 10 -n 12 -ibm 2 -ibn 3 -fbm 3 -fbn 5 -pm 2 -pn 2)

    add_test(NAME scalapack_submatrix COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 1 ./scalapack_submatrix)

    add_test(NAME parallel_scalapack_submatrix COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 ./parallel_scalapack_submatrix)
endif ()









