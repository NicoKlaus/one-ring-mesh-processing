#pragma once
#include <SimpleMesh.hpp>
#include <Vector3.hpp>
#include <vector>
#include <thrust/host_vector.h>

namespace ab {

	struct Vertex {
		Vector3 position;
		int he; //is -1 when no face contains this vertex
	};

	struct Loop {
		int he;
		bool is_border;
	};

	struct HalfEdge {
		int origin;
		int loop;

		int next;
		int prev;
		int inv;
	};

	struct HalfedgeMesh {
		std::vector<Vertex> vertices;
		std::vector<Vector3> normals;
		std::vector<HalfEdge> half_edges;
		std::vector<Loop> loops;
	};

	bool create_he_mesh_from(HalfedgeMesh& he_mesh, SimpleMesh& s_mesh);
	bool create_simple_mesh_from(SimpleMesh& s_mesh, HalfedgeMesh& he_mesh);
	void calculate_normals_he_seq(HalfedgeMesh& mesh);
}