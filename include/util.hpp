#pragma once
#include <happly.h>
#include <cuda_runtime.h>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>

namespace ab {

	void write_pointcloud(const std::string& fn, float3* points, size_t size, bool binary_mode = false);
	bool write_mesh(const SimpleMesh& mesh, const std::string& file, bool binary_mode = false);
	bool write_mesh(const HalfedgeMesh& mesh, const std::string& file, bool binary_mode = false);
	bool read_mesh(SimpleMesh& mesh, const std::string& file);
	bool read_mesh(HalfedgeMesh& mesh, const std::string& file);
}