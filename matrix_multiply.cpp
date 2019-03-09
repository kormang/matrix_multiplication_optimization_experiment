#include <ctime>
#include <cstring>
#include <iostream>
#include "common.h"

#ifndef CACHE_SIZE
#define CACHE_SIZE (32*1024)
#endif

#ifdef USE_ALIGNED
#define ALIGNED_ATTRIBUTE __attribute__ ((__aligned__(16)))
#else
#define ALIGNED_ATTRIBUTE
#endif

typedef double value_t ALIGNED_ATTRIBUTE;

typedef void (*matrix_multiply_t)(const value_t*, const value_t*, value_t*, size_t m, size_t c, size_t n);

void simple_multiply(const value_t* A, const value_t* B, value_t* C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			register value_t res = 0.0;
			for (size_t j = 0; j < c; j++) {
				res += A[i*c + j] * B[j*n + k];
			}
			C[i*n + k] = res;
		}
	}
}

// compute block size
static const size_t bs = static_cast<size_t>(sqrt(CACHE_SIZE/sizeof(value_t)/3));

void simple_block_multiply(const value_t* A, const value_t* B, value_t* C, size_t m, size_t c, size_t n) {
        #pragma omp parallel for
        for (size_t i = 0; i < m * n; ++i) {
                C[i] = 0.0;
        }

	#pragma omp parallel for
	for (size_t i = 0; i < m; i += bs) {
		for (size_t k = 0; k < n; k += bs) {
			// multiplying and summing blocks (instead of separate elements)
			for (size_t j = 0; j <= c; j += bs) {
				// multiply two blocks and add result to resulting block (instead of separate elements)
				for (size_t p = i; p < std::min(i+bs, m); ++p) {
					for (size_t r = k; r < std::min(k+bs, n); ++r) {
						register value_t res = 0.0;
						for (size_t q = j; q < std::min(j+bs, c); ++q) {
							res += A[p*c + q] * B[q*n + r];
						}
						C[p*n + r] += res;
					}
				}
			}
		}
	}

}

void multiply_ijk(const value_t *A, const value_t *B, value_t *C, size_t m, size_t c, size_t n) {
        #pragma omp parallel for
        for (size_t i = 0; i < m * n; ++i) {
                C[i] = 0.0;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < c; ++j) {
                        const value_t Aij = A[i * c + j];
                        for (size_t k = 0; k < n; ++k) {
                                C[i * n + k] += Aij * B[j * n + k];
                        }
                }
        }
}

void block_multiply_ijk_pqr(const value_t* A, const value_t* B, value_t* C, size_t m, size_t c, size_t n) {
        #pragma omp parallel for
        for (size_t i = 0; i < m*n; ++i) {
                C[i] = 0.0;
        }

	#pragma omp parallel for
	for (size_t i = 0; i < m; i += bs) {
		for (size_t j = 0; j < c; j += bs) {
			for (size_t k = 0; k < n; k += bs) {
				for (size_t p = i; p < std::min(m, i+bs); ++p) {
					for (size_t q = j; q < std::min(c, j+bs); ++q) {
						const value_t Apq = A[p*c + q];
						for (size_t r = k; r < std::min(n, k+bs); ++r) {
							C[p*n + r] += Apq * B[q*n + r];
						}
					}
				}
			}
		}
	}

}

void block_row_multiply(const value_t* A, const value_t* B, value_t* C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m*n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t i = 0; i < m; i += bs) {
		for (size_t j = 0; j < c; j += bs) {
			for (size_t p = i; p < std::min(i+bs, m); ++p) {
				for (size_t q = j; q < std::min(j+bs, c); ++q) {
					const value_t Apq = A[p*c + q];
					for (size_t r = 0; r < n; ++r) {
						C[p*n + r] += Apq * B[q*n + r];
					}
				}
			}
		}
	}
}

static const matrix_multiply_t algorithms[] = { simple_multiply, simple_block_multiply, multiply_ijk, block_multiply_ijk_pqr, block_row_multiply };
static const char* alg_names[] { "simple_multiply", "simple_block_multiply", "multiply_ijk", "block_multiply_ijk_pqr", "block_row_multiply" };

int main(int argc, char* argv[]) {

	if (argc < 6) {
		std::cerr << "Usage: matrix_multiply <algorithm-number> <m1> <cdim> <n2> [<check-result>]\n";
		return EXIT_FAILURE;
	}

	const int alg_num = atoi(argv[1]);
	const size_t m1 = (size_t)atol(argv[2]);
	const size_t cdim = (size_t)atol(argv[3]);
	const size_t n2 = (size_t)atol(argv[4]);
	const bool check_results = argc > 5 && atoi(argv[5]);

	std::cout << "Algorithm: " << alg_names[alg_num] << std::endl;
	std::cout << "First matrix dimensions: " << m1 << "x" << cdim << std::endl;
	std::cout << "Second matrix dimensions: " << cdim << "x" << n2 << std::endl;
	std::cout << "Resulting matrix dimensions: " << m1 << "x" << n2 << std::endl;
	std::cout << "Check results: " << check_results << std::endl;
	std::cout << "Align of value_t: " << alignof(value_t) << std::endl;
	std::cout << "CACHE_SIZE: " << CACHE_SIZE << std::endl;
	std::cout << "Block size: " << bs << std::endl;

	value_t* A = new value_t[m1*cdim];
	value_t* B = new value_t[cdim*n2];
	value_t* C = new value_t[m1*n2];
	value_t* D = new value_t[m1*n2];

	random_double_array(A, m1*cdim, -2.0, 2.0);
	random_double_array(B, cdim*n2, -2.0, 2.0);

	matrix_multiply_t multiply = algorithms[alg_num];

	//clock_t start = clock();
	double start = get_time_sec();

	multiply(A, B, C, m1, cdim, n2);

	//double elapsed_time = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
	double elapsed_time = get_time_sec() - start;
	std::cout << "Elapsed time: " << elapsed_time << std::endl;

	if (check_results) {
		simple_multiply(A, B, D, m1, cdim, n2);
		bool equal = are_arrays_equal(C, D, m1*n2);
		std::cout << "Results are" << (equal ? " " : " not ") << "equal\n";
	}
	return 0;

}

