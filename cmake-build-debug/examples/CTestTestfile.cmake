# CMake generated Testfile for 
# Source directory: /Users/kabicm/Projects/grid2grid/examples
# Build directory: /Users/kabicm/Projects/grid2grid/cmake-build-debug/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(square-blocks-divisible "/usr/local/bin/mpiexec" "-n" "4" "--oversubscribe" "scalapack2scalapack" "-m" "10" "-n" "10" "-ibm" "2" "-ibn" "2" "-fbm" "5" "-fbn" "5" "-pm" "2" "-pn" "2")
add_test(square-blocks-non-divisible "/usr/local/bin/mpiexec" "-n" "4" "--oversubscribe" "scalapack2scalapack" "-m" "10" "-n" "10" "-ibm" "2" "-ibn" "3" "-fbm" "3" "-fbn" "5" "-pm" "2" "-pn" "2")
add_test(non_square-blocks-non-divisible "/usr/local/bin/mpiexec" "-n" "4" "--oversubscribe" "scalapack2scalapack" "-m" "10" "-n" "12" "-ibm" "2" "-ibn" "3" "-fbm" "2" "-fbn" "6" "-pm" "2" "-pn" "2")
add_test(non-square-blocks-non-divisible "/usr/local/bin/mpiexec" "-n" "4" "--oversubscribe" "scalapack2scalapack" "-m" "10" "-n" "12" "-ibm" "2" "-ibn" "3" "-fbm" "3" "-fbn" "5" "-pm" "2" "-pn" "2")
