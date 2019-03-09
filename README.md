Test different approaches to matrix multiplication.

Following convetion is used: Matrices are C = A*B, where C has dimensions mxn, A mxc and B cxn.

First outer dimension is m, and iterating variable over that dimension is i, or p for inner block loops.

Common dimension is c, and iterating variable over that dimension is j, or q for inner block loops.

Second outer dimension is n, and iterating variable over that dimension is k, or r for inner block loops.

To check which combination of loops over i, j, and k for simple implementation is best use:
```
./test_ijk.sh
```

To test few versions of algorithm available use:
```
./test_algs.sh
```

You can edit these files to use different compiler options or send options as single argument to script:

```
./test_ijk.sh -fopenmp
./test_algs.sh "-fopenmp -DCACHE_SIZE=65536"
```

Here **-fopenmp** enables parallelisation using openmp in case you want to see how they behave in parallel.

Tested on Ubuntu 18.04, with g++ (Ubuntu 7.3.0-16ubuntu3) 7.3.0.
