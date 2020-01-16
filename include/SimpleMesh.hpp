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
	//std::vector<std::vector<int>> faces_vector; //list of indices
	std::vector<int> faces; //start indices of faces
	std::vector<int> face_indices; //continous list of face indices
	std::vector<int> face_sizes; //size of every face

	inline bool has_normals() const {
		return normals.size();
	}
};


bool read_ply(SimpleMesh &mesh,const std::string &file);
bool write_ply(const SimpleMesh &mesh,const std::string &file);

}