#pragma once
#include <SimpleMesh.hpp>
#include <Vector3.hpp>
#include <vector>

namespace ab {

	struct Vertex {
		Vector3 position;
		int he;
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
}