#! /bin/bash

g++ -O3 -DUSE_ALIGNED $1 -o bin/matrix_multiply matrix_multiply.cpp

for i in {0..4}
do
	bin/matrix_multiply $i 1000 1000 1000 1
	echo ""
done

