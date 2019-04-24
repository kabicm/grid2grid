################
#  Build test  #
################
add_executable(scalapack2scalapack "scalapack2scalapack.cpp")
target_link_libraries(scalapack2scalapack grid2grid ${DEPENDENCIES} options)

################
#  unit gtests #
################
if (MPI_FOUND)
    add_test(NAME square-blocks-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe scalapack2scalapack -m 10 -n 10 -ibm 2 -ibn 2 -fbm 5 -fbn 5 -pm 2 -pn 2)

    add_test(NAME square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe scalapack2scalapack -m 10 -n 10 -ibm 2 -ibn 3 -fbm 3 -fbn 5 -pm 2 -pn 2)

    add_test(NAME non_square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe scalapack2scalapack -m 10 -n 12 -ibm 2 -ibn 3 -fbm 2 -fbn 6 -pm 2 -pn 2)

    add_test(NAME non-square-blocks-non-divisible COMMAND ${MPIEXEC}
        ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe scalapack2scalapack -m 10 -n 12 -ibm 2 -ibn 3 -fbm 3 -fbn 5 -pm 2 -pn 2)
endif ()









