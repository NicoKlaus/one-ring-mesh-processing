#include <HalfEdgeMesh.hpp>
#include <cuda_mesh_operations.hpp>
#include <algorithm>
#include <thrust/device_vector.h>


using namespace std;

namespace ab{

	struct halfedge_sorting_tuple {
		int vertex_a, vertex_b, halfedge;

		halfedge_sorting_tuple(const int a, const int b,const int he){
			if (a > b) {
				vertex_a = b;
				vertex_b = a;
			}
			else {
				vertex_a = a;
				vertex_b = b;
			}
			halfedge = he;
		}
	};

	bool compare_by_index(const halfedge_sorting_tuple &a,const halfedge_sorting_tuple &b) {
		return (a.vertex_a < b.vertex_a) | ((a.vertex_a == b.vertex_a) & (a.vertex_b < b.vertex_b));
	}

	bool create_he_mesh_from(HalfedgeMesh& he_mesh,const SimpleMesh& s_mesh) {
		he_mesh.clear();
		//copy vertex positions
		for (size_t i = 0; i < s_mesh.vertex_count(); ++i) {
			he_mesh.vertex_positions_x.push_back(s_mesh.positions_x[i]);
			he_mesh.vertex_positions_y.push_back(s_mesh.positions_y[i]);
			he_mesh.vertex_positions_z.push_back(s_mesh.positions_z[i]);
			he_mesh.vertex_he.push_back(-1);
		}

		//copy normals
		for (size_t i = 0; i < s_mesh.normals_x.size(); ++i) {
			he_mesh.normals_x.push_back(s_mesh.normals_x[i]);
			he_mesh.normals_y.push_back(s_mesh.normals_y[i]);
			he_mesh.normals_z.push_back(s_mesh.normals_z[i]);
		}

		//process faces
		int halfedges = 0;
		for (int face = 0; face < s_mesh.faces.size();++face) {
			//add a loop
			he_mesh.loops_he.emplace_back(halfedges);
			he_mesh.loops_is_border.emplace_back(false);
			
			//add halfedges for each edge of every face
			int prev_he = halfedges+s_mesh.face_sizes[face]-1;//set first prev to the last he in the loop
			
			for (int i = 0; i < s_mesh.face_sizes[face]; ++i) {
				int vert_ind = s_mesh.face_indices[s_mesh.faces[face] + i];
				halfedges = he_mesh.add_halfedge();
				//set inv pointer to -1 to mark missing links
				he_mesh.half_edge_inv.back() = -1;
				he_mesh.half_edge_origins.back() = vert_ind;
				he_mesh.half_edge_loops.back() = he_mesh.loops_he.size()-1;
				//update prev pointer
				he_mesh.half_edge_prev.back() = prev_he;
				prev_he = halfedges-1;
				he_mesh.half_edge_next.back() =  halfedges;
				//set vertex half edge pointer
				he_mesh.vertex_he[vert_ind] = halfedges - 1;
			}
			//fix the next pointer of the last face halfe edge
			he_mesh.half_edge_next.back() = he_mesh.loops_he.back();
		}

		//create tuples for sorting, the constructor of halfeedge_tuples does the internal index sort
		vector<halfedge_sorting_tuple> halfedge_tuples;
		for (int i = 0; i < halfedges; ++i) {
			int current_origin = he_mesh.half_edge_origins[i];
			int next_origin = he_mesh.half_edge_origins[he_mesh.half_edge_next[i]];
			halfedge_tuples.emplace_back(current_origin, next_origin, i);
		}
		
		sort(halfedge_tuples.begin(), halfedge_tuples.end(), compare_by_index);
		
		//link inv pointers
		{
			halfedge_sorting_tuple *last = &halfedge_tuples.front();
			for (int i = 1; i < halfedge_tuples.size(); ++i) {
				halfedge_sorting_tuple *he = &halfedge_tuples[i];
				if ((last != nullptr) && (he->vertex_a == last->vertex_a) && (he->vertex_b == last->vertex_b)) {
					he_mesh.half_edge_inv[he->halfedge] = last->halfedge;
					he_mesh.half_edge_inv[last->halfedge] = he->halfedge;
					last = nullptr;
					continue;
				}
				last = he;
			}
		}

		//boundary loops halfedges

		for (int i = 0; i < halfedges; ++i) {
			if (he_mesh.half_edge_inv[i] == -1) {
				halfedges = he_mesh.add_halfedge();
				he_mesh.half_edge_inv[i] = halfedges-1;
				he_mesh.half_edge_inv.back() = i;
				he_mesh.half_edge_origins.back() = he_mesh.half_edge_origins[he_mesh.half_edge_next[i]];
				he_mesh.half_edge_next.back() = -1;
				he_mesh.half_edge_prev.back() = -1;
				he_mesh.half_edge_loops.back() = -1;
			}
		}

		//remaining halfedges without a next or prev pointer are part of boundary loops
		for (int i = 0; i < halfedges; ++i) {
			if (he_mesh.half_edge_next[i] == -1) {
				//create new boundary halfedge
				//circulate trough vertex halfedges
				
				int next = i;
				while (he_mesh.half_edge_prev[he_mesh.half_edge_inv[next]] != -1) {
					next = he_mesh.half_edge_prev[he_mesh.half_edge_inv[next]];
				}
				he_mesh.half_edge_next[i] = he_mesh.half_edge_inv[next];
			}
		}

		//complete boundary loops and set prev pointers
		//boundary halfedges have loop set to -1 at this state

		for (int i = 0; i < halfedges; ++i) {
			if (he_mesh.half_edge_loops[i] == -1) {
				int loops_size = he_mesh.add_loop();
				int lo = loops_size-1;
				he_mesh.loops_he[lo] = i;
				he_mesh.loops_is_border[lo] = true;
				int he = i;
				while (he_mesh.half_edge_loops[he] == -1) {
					he_mesh.half_edge_loops[he] = lo;
					he_mesh.half_edge_prev[he_mesh.half_edge_next[he]] = he;
					he = he_mesh.half_edge_next[he];
				}
			}
		}
		return true;
	}

	bool create_simple_mesh_from(SimpleMesh& s_mesh,const HalfedgeMesh& he_mesh)
	{
		//reset arrays
		s_mesh.clear();

		s_mesh.normals_x = he_mesh.normals_x;
		s_mesh.normals_y = he_mesh.normals_y;
		s_mesh.normals_z = he_mesh.normals_z;

		s_mesh.positions_x = he_mesh.vertex_positions_x;
		s_mesh.positions_y = he_mesh.vertex_positions_y;
		s_mesh.positions_z = he_mesh.vertex_positions_z;

		//construct face list
		for (int i=0; i < he_mesh.loops_size(); ++i) {
			if (he_mesh.loops_is_border[i]) {
				continue;
			}

			int origin = he_mesh.loops_he[i];
			int next = origin;
			//start a new face
			s_mesh.faces.emplace_back(s_mesh.face_indices.size());
			
			int size = 0;
			do {
				s_mesh.face_indices.emplace_back(he_mesh.half_edge_origins[next]);
				next = he_mesh.half_edge_next[next];
				++size;
			} while (next != origin);
			s_mesh.face_sizes.emplace_back(size);
		}
		return true;
	}

}