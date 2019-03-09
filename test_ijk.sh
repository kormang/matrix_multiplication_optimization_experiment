#! /bin/bash

g++ -O3 $1 -o bin/matrix_multiply_ijk matrix_multiply_ijk.cpp

for i in {0..5}
do
	bin/matrix_multiply_ijk $i 1000 1000 1000 0
	echo ""
done

