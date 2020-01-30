#include <mesh_normal_generator.hpp>
#include <cpu_mesh_operations.hpp>
#include <cuda_mesh_operations.hpp>

using namespace std;

namespace ab {
	mesh_normal_generator::mesh_normal_generator(SimpleMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks)
	{
		mesh_pointer = mesh;
		proc_mode = PM_SCATTER;
		proc_dev = dev;
		this->threads = threads;
		this->blocks = blocks;
	}

	mesh_normal_generator::mesh_normal_generator(HalfedgeMesh* mesh, ProcessingDevice dev, size_t threads, size_t blocks)
	{
		mesh_pointer = mesh;
		proc_mode = PM_GATHER;
		proc_dev = dev;
		this->threads = threads;
		this->blocks = blocks;
	}

	mesh_normal_generator::~mesh_normal_generator()
	{
	}

	void mesh_normal_generator::operator()()
	{
		switch (proc_mode) {
		case PM_SCATTER: {
			SimpleMesh* mesh = reinterpret_cast<SimpleMesh*>(mesh_pointer);
			if (proc_dev == PD_CPU) {
				normals_by_area_weight_sm_cpu(mesh, threads);
			}
			else if(proc_dev == PD_CUDA) {

			}
		}
		case PM_GATHER: {
			HalfedgeMesh* mesh = reinterpret_cast<HalfedgeMesh*>(mesh_pointer);
		}
		}
	}

}