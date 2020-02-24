#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <onering_base.hpp>

namespace ab {

struct SimpleMesh
{
	//vertice attribute arrays
	attribute_vector<float3> positions; //contains the vertex positions
	attribute_vector<float3> normals; //contains vertex normals
	//connectivity
	//std::vector<std::vector<int>> faces_vector; //list of indices
	attribute_vector<int> faces; //elements point to the start index  of a face in face_indices
	attribute_vector<int> face_indices; //list of face indices, a face begins at face_indices[faces[i]] and ends at face_indices[faces[i]+face_size[i]-1]
	attribute_vector<int> face_sizes; //size of every face

	inline bool has_normals() const {
		return normals.size();
	}
};

inline int vertex_count_of(const SimpleMesh& mesh) {
	return mesh.positions.size();
}


inline int face_count_of(const SimpleMesh& mesh) {
	return mesh.faces.size();
}

inline size_t in_memory_size_of(const SimpleMesh& mesh) {
	return sizeof(float3) * (mesh.positions.size() + mesh.normals.size()) +
		sizeof(int) * (mesh.faces.size() + mesh.face_indices.size() + mesh.face_sizes.size());
}

bool write_ply(const SimpleMesh &mesh,const std::string &file);

}