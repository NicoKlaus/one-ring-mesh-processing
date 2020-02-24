#pragma once
#include <device_launch_parameters.h>

namespace ab {
	template <typename T>
	__host__ void optimal_configuration(int& blocks, int& threads, const T& kernel) {
		int grid_size = 0;
		int block_size = 64;
		cudaDeviceGetAttribute(&grid_size, cudaDevAttrMultiProcessorCount,0);

		//cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, 0, 256);
		blocks = grid_size;
		threads = block_size;
	}

	__host__ size_t cuda_elapsed_time(cudaEvent_t& cu_start, cudaEvent_t& cu_stop) {
		float exec_time;
		cudaEventElapsedTime(&exec_time, cu_start, cu_stop);
		return static_cast<size_t>(exec_time * 1000000);
	}

	

#if __CUDA_ARCH__ < 600
	__device__ int atomicAdd(int* address, int val)
	{
		int old = *address, assumed;

		do {
			assumed = old;
			old = atomicCAS(address, assumed, val + assumed);
		} while (assumed != old);

		return old;
	}
#endif

#if __CUDA_ARCH__ < 600
	__device__ double atomicAdd(double* address, double val)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;

		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));

			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);

		return __longlong_as_double(old);
	}
#endif

#if __CUDA_ARCH__ < 600
	__device__ float atomicAdd(float* address, float val)
	{
		unsigned int* address_as_ull =
			(unsigned int*)address;
		unsigned int old = *address_as_ull, assumed;

		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val +
					__int_as_float(assumed)));

			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);

		return __int_as_float(old);
	}
#endif
}