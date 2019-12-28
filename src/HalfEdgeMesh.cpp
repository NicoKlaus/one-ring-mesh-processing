#include <HalfEdgeMesh.hpp>
#include <algorithm>

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

	bool create_he_mesh_from(HalfedgeMesh& he_mesh, SimpleMesh& s_mesh) {
		he_mesh.vertices.resize(0);
		he_mesh.half_edges.resize(0);
		//copy vertex positions
		Vertex v;
		for (size_t i = 0; i < s_mesh.positions.size(); ++i) {
			v.position = s_mesh.positions[i];
			v.he = -1;
			he_mesh.vertices.push_back(v);
		}

		//copy normals
		he_mesh.normals = s_mesh.normals;

		auto &halfedges = he_mesh.half_edges;
		auto &loops = he_mesh.loops;
		auto &vertices = he_mesh.vertices;

		//process faces
		for (auto face : s_mesh.faces) {
			//add a loop
			loops.emplace_back();
			Loop &loop = he_mesh.loops.back();
			loop.is_border = false;
			loop.he = halfedges.size();

			//add halfedges for each edge of every face
			int prev_he = halfedges.size()+face.size()-1;//set first prev to the last he in the loop
			for (auto vert_ind : face) {
				he_mesh.half_edges.emplace_back();
				HalfEdge &he = he_mesh.half_edges.back();
				//set inv pointer to -1 to mark missing links
				he.inv = -1;
				he.origin = vert_ind;
				he.loop = loops.size()-1;
				//update prev pointer
				he.prev = prev_he;
				prev_he = halfedges.size()-1;
				he.next = halfedges.size();
				//set vertex half edge pointer
				vertices[vert_ind].he = halfedges.size() - 1;
			}
			//fix the next pointer of the last face halfe edge
			he_mesh.half_edges.back().next = he_mesh.loops.back().he;
		}

		//create tuples for sorting, the constructor of halfeedge_tuples does the internal index sort
		vector<halfedge_sorting_tuple> halfedge_tuples;
		for (int i = 0; i < halfedges.size(); ++i) {
			HalfEdge &current = halfedges[i];
			HalfEdge &next = halfedges[current.next];
			halfedge_tuples.emplace_back(current.origin, next.origin, i);
		}
		
		sort(halfedge_tuples.begin(), halfedge_tuples.end(), compare_by_index);
		
		//link inv pointers
		{
			halfedge_sorting_tuple *last = &halfedge_tuples.front();
			for (int i = 1; i < halfedge_tuples.size(); ++i) {
				halfedge_sorting_tuple *he = &halfedge_tuples[i];
				if ((last != nullptr) && (he->vertex_a == last->vertex_a) && (he->vertex_b == last->vertex_b)) {
					halfedges[he->halfedge].inv = last->halfedge;
					halfedges[last->halfedge].inv = he->halfedge;
					last = nullptr;
					continue;
				}
				last = he;
			}
		}

		//boundary loops halfedges

		for (int i = 0; i < halfedges.size(); ++i) {
			HalfEdge &halfedge = halfedges[i];
			if (halfedges[i].inv == -1) {
				halfedges.emplace_back();
				HalfEdge &boundary_he = halfedges.back();
				halfedge.inv = halfedges.size()-1;
				boundary_he.inv = i;
				boundary_he.origin = halfedges[halfedge.next].origin;
				boundary_he.next = -1;
				boundary_he.prev = -1;
				boundary_he.loop = -1;
			}
		}

		//remaining halfedges without a next or prev pointer are part of boundary loops
		for (int i = 0; i < halfedges.size(); ++i) {
			if (halfedges[i].next == -1) {
				//create new boundary halfedge
				//circulate trough vertex halfedges
				
				int next = i;
				while (halfedges[halfedges[next].inv].prev != -1) {
					next = halfedges[halfedges[next].inv].prev;
				}
				halfedges[i].next = halfedges[next].inv;
			}
		}

		//complete boundary loops and set prev pointers
		//boundary halfedges have loop set to -1 at this state

		for (int i = 0; i < halfedges.size(); ++i) {
			if (halfedges[i].loop == -1) {
				loops.emplace_back();
				int lo = loops.size()-1;
				loops[lo].he = i;
				loops[lo].is_border = true;
				int he = i;
				while (halfedges[he].loop == -1) {
					halfedges[he].loop = lo;
					halfedges[halfedges[he].next].prev = he;
					he = halfedges[he].next;
				}
			}
		}
		return true;
	}

	bool create_simple_mesh_from(SimpleMesh & s_mesh, HalfedgeMesh & he_mesh)
	{
		s_mesh.positions.resize(0);
		s_mesh.faces.resize(0);
		s_mesh.normals.resize(0);
		for (auto vertice : he_mesh.vertices) {
			s_mesh.positions.emplace_back(vertice.position);
		}
		for (int i=0; i < he_mesh.loops.size(); ++i) {
			if (he_mesh.loops[i].is_border) {
				continue;
			}

			int origin = he_mesh.loops[i].he;
			int next = origin;
			auto &faces = s_mesh.faces;
			faces.emplace_back();
			auto &face = faces.back();
			do {
				face.emplace_back(he_mesh.half_edges[next].origin);
				next = he_mesh.half_edges[next].next;
			} while (next != origin);
			
		}
		return false;
	}

	void calculate_normals_he_seq(HalfedgeMesh &mesh) {
		//initialize normal array
		mesh.normals.resize(mesh.vertices.size());
		//calculate normal without weight
		for (int i = 0; i < mesh.vertices.size();++i) {
			auto& vert = mesh.vertices[i];
			if (vert.he == -1) {
				continue;
			}
			int he = vert.he;
			Vector3 normal{ 0.f,0.f,0.f };
			do {
				HalfEdge& halfedge = mesh.half_edges[he];
				Vector3 a = vert.position;
				Vector3 b = mesh.vertices[mesh.half_edges[halfedge.next].origin].position;
				normal += cross(a, b);
				he = halfedge.next;
			} while (he != vert.he);
			normal = normalized(normal);
			mesh.normals[i] = normal;
		}
	}

	void calculate_normals_he_parallel(HalfedgeMesh *mesh) {

	}

	__global__ void kernel_calculate_normals(Vertex* vertices, HalfEdge* halfedges, Vector3 *normals,unsigned vertice_count) {

	}
}