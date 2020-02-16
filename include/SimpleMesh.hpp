#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace ab {

struct SimpleMesh
{
	//vertice attribute arrays
	std::vector<float> positions_x;
	std::vector<float> positions_y;
	std::vector<float> positions_z;
	std::vector<float> normals_x;
	std::vector<float> normals_y;
	std::vector<float> normals_z;
	//connectivity
	//std::vector<std::vector<int>> faces_vector; //list of indices
	std::vector<int> faces; //elements point to the start index  of a face in face_indices
	std::vector<int> face_indices; //list of face indices, a face begins at face_indices[faces[i]] and ends at face_indices[faces[i]+face_size[i]-1]
	std::vector<int> face_sizes; //size of every face

	inline bool has_normals() const {
		return normals_x.size();
	}

	inline int vertex_count() const {
		return positions_x.size();
	}

	inline void clear_normals() {
		normals_x.resize(0);
		normals_y.resize(0);
		normals_z.resize(0);
	}

	inline void resize_normals(size_t s) {
		normals_x.resize(s);
		normals_y.resize(s);
		normals_z.resize(s);
	}

	inline void clear() {
		positions_x.resize(0);
		positions_y.resize(0);
		positions_z.resize(0);
		faces.resize(0);
		face_indices.resize(0);
		face_sizes.resize(0);
		resize_normals(0);
	}
};


}