#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace ab {

struct SimpleMesh
{
	//vertice attribute arrays
	std::vector<float3> positions;
	std::vector<float3> normals;
	//connectivity
	std::vector<std::vector<int>> faces; //list of indices

	inline bool has_normals() const {
		return normals.size();
	}
};


bool read_ply(SimpleMesh &mesh,const std::string &file);
bool write_ply(const SimpleMesh &mesh,const std::string &file);

}