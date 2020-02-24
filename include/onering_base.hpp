#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace ab {
	template <class T> using attribute_vector = std::vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
}