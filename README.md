
# grid2grid

This is a library that transforms a matrix between two arbitrary grid-like data layouts. By layout, we mean the way in which a matrix is distributed over MPI ranks.

For example, imagine a matrix split into four blocks, such that each block resides on a different rank as shown below:

```
+---------------------+
|       |             |
|       |             |
|   0   |      1      |
|       |             |
|       |             |
+---------------------+
|       |             |
|   2   |      3      |
|       |             |
+---------------------+
```

 Assume that we want to redistribute the matrix, to achieve the following data layout:
 
```
+---------------------+
|   0   |    2    | 1 |
|       |         |   |
+---------------------+
|   1   |    0    | 1 |
|       |         |   |
+---------------------+
|       |         |   |
|   2   |    2    | 0 |
|       |         |   |
+---------------------+
```

This is exaclty what this library is doing!

## Features:

- initial and final layouts do not have to be block-cyclic, but any grid-like data layout.
- blocks do not have to be all the same dimensions, but can have different dimensions.
- a rank might own arbitrary number of blocks
- the type of entries in a matrix can be arbitrary
- blocks that belong to the same rank do not have to be consecutive in memory
- each block can have an arbitrary stride
- OpenMP support

## Performance:

When tested against ScaLAPACK (as provided by Intel MKL), this library outperforms the MKL's `pdgemr2d` routine that transforms between two block-cyclic data layouts by up to 50\%, as shown below:

<p align="center"><img src="https://github.com/kabicm/grid2grid/blob/master/docs/performance.svg" width="80%"></p>

## Algorithm

The pipeline of the algorithm is roughly the following:
- Find the intersections of the initial and final grids (called grid cover) in one pass, which decomposes initial blocks into smaller blocks.
- Sort decomposed initial blocks based on the rank to which they should be sent. Thus, if some ranks should exchange more than one block, they will do it within a single message.
- Copy decomposed intial blocks from a local storage (with arbitrary local data layout) to a temporary MPI buffer.
- Perform MPI_Alltoallv.
- Use the computed grid cover to decompose blocks from the final grid into smaller blocks.
- Sort decomposed final blocks based on the receiver rank.
- All smaller blocks that are received are copied from a temporary MPI buffer to the local storage (with arbitrary local data layout).

## Building and Installing

Assuming that you want to use the `gcc 8` compiler and `OpenMP`, you can build the project as follows:
```
# clone the repo
git clone https://github.com/kabicm/grid2grid
cd grid2grid
mkdir build
cd build

# build
CC=gcc-8 CXX=g++-8 cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=TRUE ..

# compile
make -j 4
```

## Example

To start with, there is a small example that transforms the matrix between two block-cyclic layouts. This example can be run from `build` directory as follows:
```
mpirun --oversubscribe -np 4 ./examples/scalapack2scalapack -m 10 -n 10 -ibm 2 -ibn 3 -fbm 3 -fbn 5 -pm 2 -pn 2
```
Where flags have the following meaning:
- `(m, n)`: Dimensions of matrix that we want to change the layout for.
- `(ibm, ibn)`: Dimensions of initial blocks, determining the initial block-cyclic distribution.
- `(fbm, fbn)`: Dimensions of final blocks, determining the final block-cyclic distribution that we want to reach.
- `(pm, pn)`: Processor grid, determining the processor decomposition for the block-cyclic distribution.

## Arbitrary Grid-Like Data Layouts

To transform between two arbitrary grid-like data layouts, we need to construct two `grid_layout` objects, one describing the initial layout and one describing the final layout. After that, we just invoke:
```
grid2grid::grid_layout initial_layout(...);
grid2grid::grid_layout final_layout(...);
grid2grid::transform(initial_layout, final_layout, MPI_COMM_WORLD);
```

In order to create a grid_layout object, we need to provide the following information:
- `grid2D`: small struct describing the grid, basically 2 vectors, one describing where rows are split and one describing where columns are split in a grid.
- `owners`: a matrix that specifies the owner (i.e. the rank) of each block in `grid2D`.
- `local_blocks`: a vector of blocks that current rank owns. Each block is defined with a pointer to the beginning of that block in the local memory of current rank, a stride (by default equal to the number of rows of a block) and dimensions.

This is done for `ScaLAPACK` block-cyclic data layout, which can serve as an example of how `grid_layout` object can be created.

## Author
Marko Kabic (marko.kabic@cscs.ch)
