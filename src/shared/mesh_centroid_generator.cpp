#include <mesh_centroid_generator.hpp>
#include <cpu_mesh_operations.hpp>
#include <cuda_mesh_operations.hpp>

using namespace std;

namespace ab {
	mesh_centroid_generator::mesh_centroid_generator(SimpleMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks)
	{
		mesh_pointer = mesh;
		proc_mode = PM_SCATTER;
		proc_dev = dev;
		this->threads = threads;
		this->blocks = blocks;
	}

	mesh_centroid_generator::mesh_centroid_generator(HalfedgeMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks)
	{
		mesh_pointer = mesh;
		proc_mode = PM_GATHER;
		proc_dev = dev;
		this->threads = threads;
		this->blocks = blocks;
	}

	mesh_centroid_generator::~mesh_centroid_generator()
	{
	}

	void mesh_centroid_generator::operator()()
	{
		switch (proc_mode) {
		case PM_SCATTER: {
			SimpleMesh* mesh = reinterpret_cast<SimpleMesh*>(mesh_pointer);
			if (proc_dev == PD_CPU) {
				centroids_sm_cpu(mesh, centroids_x, centroids_y, centroids_z, threads, timings);
			}
			else if (proc_dev == PD_CUDA) {
				centroids_sm_cuda(mesh,centroids_x, centroids_y, centroids_z, threads, blocks, timings);
			}
			break;
		}
		case PM_GATHER: {
			HalfedgeMesh* mesh = reinterpret_cast<HalfedgeMesh*>(mesh_pointer);
			if (proc_dev == PD_CPU) {
				centroids_he_cpu(mesh, centroids_x, centroids_y, centroids_z, threads, timings);
			}
			else if (proc_dev == PD_CUDA) {
				centroids_he_cuda(mesh, centroids_x, centroids_y, centroids_z, threads, blocks, timings);
			}
			break;
		}
		}
	}

}