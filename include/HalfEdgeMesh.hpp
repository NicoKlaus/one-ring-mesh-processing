#pragma once
#include <SimpleMesh.hpp>
#include <vector>
#include <onering_base.hpp>


namespace ab {

	struct Vertex {
		float3 position;
		int he; //is -1 when no face contains this vertex
	};

	struct Loop {
		int he;
		bool is_border;
	};

	struct alignas(16) ReducedHalfEdge {
		int origin;
		int loop;
		int next;
		int inv;
	};

	struct HalfEdge {
		int origin;
		int loop;

		int next;
		int prev;
		int inv;
	};

	struct HalfedgeMesh {
		attribute_vector<Vertex> vertices;
		attribute_vector<float3> normals;
		attribute_vector<HalfEdge> half_edges;
		attribute_vector<ReducedHalfEdge> reduced_half_edges;
		attribute_vector<Loop> loops;
	};

	inline void create_reduced_halfedges(HalfedgeMesh& mesh) {
		mesh.reduced_half_edges.resize(mesh.half_edges.size());
		for (int i = 0; i < mesh.reduced_half_edges.size(); ++i) {
			ReducedHalfEdge halfedge;
			halfedge.inv = mesh.half_edges[i].inv;
			halfedge.next = mesh.half_edges[i].next;
			halfedge.loop = mesh.half_edges[i].loop;
			halfedge.origin = mesh.half_edges[i].origin;
			mesh.reduced_half_edges[i] = halfedge;
		}
	}

	inline int vertex_count_of(const HalfedgeMesh& mesh) {
		return mesh.vertices.size();
	}

	inline int face_count_of(const HalfedgeMesh& mesh) {
		int faces = 0;
		for (auto loop : mesh.loops) {
			faces += !loop.is_border;
		}
		return faces;
	}

	inline size_t in_memory_size_of(const HalfedgeMesh& mesh) {
		return sizeof(HalfEdge) * mesh.half_edges.size() +
			sizeof(Vertex) * mesh.vertices.size() +
			sizeof(Loop) * mesh.loops.size() +
			sizeof(float3) * mesh.normals.size();
	}

	bool write_ply(const HalfedgeMesh& mesh, const std::string& file);

	bool create_he_mesh_from(HalfedgeMesh& he_mesh,const SimpleMesh& s_mesh);
	bool create_simple_mesh_from(SimpleMesh& s_mesh,const HalfedgeMesh& he_mesh);
	void calculate_normals_he_seq(HalfedgeMesh& mesh);
}