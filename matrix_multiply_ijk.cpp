#include "common.h"
#include <cstring>
#include <ctime>
#include <iostream>

typedef void (*matrix_multiply_t)(const double *, const double *, double *, size_t m, size_t c, size_t n);

void multiply_ijk(const double *A, const double *B, double *C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m * n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < c; ++j) {
			const double Aij = A[i * c + j];
			for (size_t k = 0; k < n; ++k) {
				C[i * n + k] += Aij * B[j * n + k];
			}
		}
	}
}

void multiply_ikj(const double *A, const double *B, double *C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m * n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			double res = 0.0;
			for (size_t j = 0; j < c; ++j) {
				res += A[i * c + j] * B[j * n + k];
			}
			C[i * n + k] = res;
		}
	}
}

void multiply_jik(const double *A, const double *B, double *C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m * n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t j = 0; j < c; ++j) {
		for (size_t i = 0; i < m; ++i) {
			const double Aij = A[i * c + j];
			for (size_t k = 0; k < n; ++k) {
				C[i * n + k] += Aij * B[j * n + k];
			}
		}
	}
}

void multiply_jki(const double *A, const double *B, double *C, size_t m, #pragma omp parallel for
	for (size_t i = 0; i < m * n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t j = 0; j < c; ++j) {
		for (size_t k = 0; k < n; ++k) {
			const double Bjk = B[j * n + k];
			for (size_t i = 0; i < m; ++i) {
				C[i * n + k] += A[i * c + j] * Bjk;
			}
		}
	}
}

void multiply_kij(const double *A, const double *B, double *C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t k = 0; k < n; ++k) {
		for (size_t i = 0; i < m; ++i) {
			double res = 0.0;
			for (size_t j = 0; j < c; ++j) {
				res += A[i * c + j] * B[j * n + k];
			}
			C[i * n + k] = res;
		}
	}
}

void multiply_kji(const double *A, const double *B, double *C, size_t m, size_t c, size_t n) {
	#pragma omp parallel for
	for (size_t i = 0; i < m * n; ++i) {
		C[i] = 0.0;
	}

	#pragma omp parallel for
	for (size_t k = 0; k < n; ++k) {
		for (size_t j = 0; j < c; ++j) {
			const double Bjk = B[j * n + k];
			for (size_t i = 0; i < m; ++i) {
				C[i * n + k] += A[i * c + j] * Bjk;
			}
		}
	}
}

static const matrix_multiply_t algorithms[] = { multiply_ijk, multiply_ikj, multiply_jik, multiply_jki, multiply_kij, multiply_kji };
static const char *alg_names[] = { "multiply_ijk", "multiply_ikj", "multiply_jik", "multiply_jki", "multiply_kij", "multiply_kji" };

int main(int argc, char *argv[]) {
	if (argc < 6) {
		std::cerr << "Usage: matrix_multiply <algorithm-number> <m1>  <cdim> <n2>  [<check-result>]\n";
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

	double *A = new double[m1 * cdim];
	double *B = new double[cdim * n2];
	double *C = new double[m1 * n2];
	double *D = new double[m1 * n2];

	random_double_array(A, m1 * cdim, -2.0, 2.0);
	random_double_array(B, cdim * n2, -2.0, 2.0);

	matrix_multiply_t multiply = algorithms[alg_num];

	// clock_t start = clock();
	double start = get_time_sec();

	multiply(A, B, C, m1, cdim, n2);

	// double elapsed_time = static_cast<double>(clock() - start) /
	// CLOCKS_PER_SEC;
	double elapsed_time = get_time_sec() - start;
	std::cout << "Elapsed time: " << elapsed_time << std::endl;

	if (check_results) {
		multiply_ijk(A, B, D, m1, cdim, n2);
		bool equal = are_arrays_equal(C, D, m1 * n2);
		std::cout << "Results are" << (equal ? " " : " not ") << "equal\n";
	}
	return 0;
}
