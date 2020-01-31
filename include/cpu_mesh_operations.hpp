#pragma once
#include <HalfEdgeMesh.hpp>
#include <cuda_mesh_operations.hpp> //nedded for cross3df
#include <thread>
#include <timing_struct.hpp>

namespace ab {
	
	void normals_by_area_weight_he_cpu(HalfedgeMesh* mesh, int threads=8, timing_struct& timing = timing_struct());
	void normals_by_area_weight_sm_cpu(SimpleMesh* mesh, int threads=8, timing_struct& timing = timing_struct());
}