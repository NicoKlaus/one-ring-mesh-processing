#pragma once
#include <SimpleMesh.hpp>
#include <vector>


namespace ab {
	/*normal halfedge mesh structure*/
	/*
	struct Vertex {
		float3 position;
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
		std::vector<float3> normals;
		std::vector<HalfEdge> half_edges;
		std::vector<Loop> loops;
	};
	*/

	struct HalfedgeMesh {
		std::vector<float> vertex_positions_x;
		std::vector<float> vertex_positions_y;
		std::vector<float> vertex_positions_z;
		std::vector<int> vertex_he;
		std::vector<float> normals_x;
		std::vector<float> normals_y;
		std::vector<float> normals_z;
		std::vector<int> half_edge_origins;
		std::vector<int> half_edge_loops;
		std::vector<int> half_edge_next;
		std::vector<int> half_edge_prev;
		std::vector<int> half_edge_inv;
		std::vector<int> loops_he;
		std::vector<char> loops_is_border;


		inline int loops_size() const {
			return loops_he.size();
		}

		inline int add_halfedge() {
			half_edge_inv.emplace_back();
			half_edge_loops.emplace_back();
			half_edge_next.emplace_back();
			half_edge_origins.emplace_back();
			half_edge_prev.emplace_back();
			return half_edge_inv.size();
		}

		inline int add_loop() {
			loops_he.emplace_back();
			loops_is_border.emplace_back();
			return loops_he.size();
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

		inline int vertex_count() {
			return vertex_positions_x.size();
		}

		inline void clear() {
			clear_normals();
			vertex_positions_x.resize(0);
			vertex_positions_y.resize(0);
			vertex_positions_z.resize(0);
			vertex_he.resize(0);
			half_edge_origins.resize(0);
			half_edge_loops.resize(0);
			half_edge_next.resize(0);
			half_edge_prev.resize(0);
			half_edge_inv.resize(0);
			loops_he.resize(0);
			loops_is_border.resize(0);
		}
	};

	bool create_he_mesh_from(HalfedgeMesh& he_mesh,const SimpleMesh& s_mesh);
	bool create_simple_mesh_from(SimpleMesh& s_mesh,const HalfedgeMesh& he_mesh);
}