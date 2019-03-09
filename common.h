#include <random>
#include <sys/time.h>

double get_time_sec() {
	struct timeval ct;
        gettimeofday(&ct, 0);
        /* return time in seconds */
        return (static_cast<double>(ct.tv_sec) + static_cast<double>(ct.tv_usec) / 1E6);
}

void random_double_array(double* array, size_t size, double a, double b) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(a, b);
    for (size_t n = 0; n < size; ++n) {
         array[n] = dis(gen);
    }
}

bool are_arrays_equal(double* a1, double* a2, size_t size) {
	for (size_t n = 0; n < size; ++n) {
		register double r = (a1[n] - a2[n]);
		if (r < -0.00001 || r > 0.00001) {
			return false;
		}
	}
	return true;
}

bool are_all_zeros(double* array, size_t size) {
	for (size_t n = 0; n < size; ++n) {
		if (array[n] != 0.0) {
			return false;
		}
	}
	return true;
}
