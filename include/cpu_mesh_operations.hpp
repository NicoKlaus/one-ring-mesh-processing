#pragma once
#include <HalfEdgeMesh.hpp>
#include <cuda_mesh_operations.hpp> //nedded for cross3df
#include <thread>

namespace ab {
	
	void normals_by_area_weight_cpu(HalfedgeMesh* mesh, int threads);
}