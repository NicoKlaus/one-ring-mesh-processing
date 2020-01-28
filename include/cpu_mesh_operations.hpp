#pragma once
#include <HalfEdgeMesh.hpp>
#include <cuda_mesh_operations.hpp> //nedded for cross3df
#include <thread>

namespace ab {
	
	void normals_by_area_weight_he_cpu(HalfedgeMesh* mesh, int threads);
	void normals_by_area_weight_sm_cpu(SimpleMesh* mesh, int threads);
}