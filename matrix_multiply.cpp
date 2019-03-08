#include <ctime>
#include <cstring>
#include <iostream>
#include <omp.h>
#include "common.h"

typedef void (*matrix_multiply_t)(double*, double*, double*, size_t m, size_t c, size_t n);

void simple_multiply(double* A, double* B, double* C, size_t m, size_t c, size_t n) {

	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			register double res = 0.0;
			for (size_t k = 0; k < c; k++) {
				res += A[i*c + k] * B[k*n + j];
			}
			C[i*n + j] = res;
		}
	}
}

#define CACHE_SIZE (32*1024)
// compute block size
const size_t bs = static_cast<size_t>(sqrt(CACHE_SIZE/sizeof(double)/3));
void block_multiply(double* A, double* B, double* C, size_t m, size_t c, size_t n) {
	for (size_t i = 0; i < m; i += bs) {
		for (size_t j = 0; j < n; j += bs) {	
			// initialization of block (instead of element)
			for (size_t p = 0; p < std::min(bs, m - i); p += 1) {
				for (size_t q = 0; q < std::min(bs, n - j); q += 1) {
					C[(i+p)*m + j + q] = 0.0;
				}
			}
			// multiplying and summing blocks (instead of separate elements)
			for (size_t k = 0; k <= c; k += bs) {
				// multiply two blocks and add result to resulting block (instead of separate elements)
				for (size_t p = 0; p < std::min(bs, m - i); p += 1) {
					for (size_t q = 0; q < std::min(bs, n - j); q += 1) {
						register double res = 0.0;
						for (size_t r = 0; r < std::min(bs, c - k); r += 1) {
							res += A[(i+p)*c + k + r] * B[(k+r)*n + j + q];
						}
						C[(i+p)*n + j + q] += res;
					}
				}
			}
		}
	}

}

int main() {
	const size_t m = 1734;
	const size_t n = 1347;

	double* A = new double[m*n];
	double* B = new double[m*n];
	double* C = new double[m*m];
	double* D = new double[m*m];
	
	random_double_array(A, m*n, -2.0, 2.0);
	random_double_array(B, m*n, -2.0, 2.0);

	matrix_multiply_t multiply = block_multiply;

	clock_t start = clock();

	multiply(A, B, C, m, n, m);

	double elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
	std::cout << "Elapsed time: " << elapsedTime << std::endl;

	simple_multiply(A, B, D, m, n, m);
	bool equal = are_arrays_equal(C, D, m*m);
	std::cout << "Results are" << (equal ? " " : " not ") << "equal\n";
	//std::cout << "All are " << (are_all_zeros(C, m*m) ? "" : "not ") << "zeros\n";

	return 0;

}
