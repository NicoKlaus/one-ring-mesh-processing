#include <happly.h>
#include <cuda_runtime.h>

namespace ab {

	void write_pointcloud(const std::string& fn, float3* points, size_t size);
}