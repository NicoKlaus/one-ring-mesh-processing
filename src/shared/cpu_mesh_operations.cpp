#include <cpu_mesh_operations.hpp>
#include <intrin.h>
#include <utility>
#include <algorithm>
#include <mutex>

using namespace std;

namespace ab {

	float atomic_add(float* address, float val)
	{
		volatile long* address_as_ull =
			(long*)address;
		long old = *address_as_ull, assumed;

		do {
			assumed = old;
			float new_value = reinterpret_cast<float&>(assumed) + val;
			old = _InterlockedCompareExchange(address_as_ull, reinterpret_cast<long&>(new_value), assumed);
		} while (assumed != old);

		return reinterpret_cast<float&>(old);
	}

	float atomic_add(int* address, int val)
	{
		volatile long* address_as_ull =
			(long*)address;
		long old = *address_as_ull, assumed;

		do {
			assumed = old;
			int new_value = reinterpret_cast<int&>(assumed) + val;
			old = _InterlockedCompareExchange(address_as_ull, reinterpret_cast<long&>(new_value),assumed);
		} while (assumed != old);

		return  reinterpret_cast<int&>(old);
	}

	void cpu_kernel_normalize_vectors(float3* vec, unsigned size, int stride, int offset) {
		for (int i = offset; i < size; i += stride) {
			vec[i] = normalized(vec[i]);
		}
	}


	void cpu_kernel_divide(float3* vec, int* div, unsigned vec_size, int stride, int offset) {
		for (int i = offset; i < vec_size; i += stride) {
			float fdiv = 1.f / static_cast<float>(div[i]);
			vec[i].x *= fdiv;
			vec[i].y *= fdiv;
			vec[i].z *= fdiv;
		}
	}

	void cpu_kernel_normals_gather(Vertex* vertices, HalfEdge* half_edges, Loop* loops, float3* normals, unsigned vertice_count,int stride,int offset) {
		//calculate normals
		for (int i = offset; i < vertice_count; i += stride) {
			auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}

			float3 normal{ 0.f,0.f,0.f };

			int he = vert.he;
			do {//for every neighbor
				HalfEdge& halfedge = half_edges[he];
				//skip boundary loops
				if (loops[halfedge.loop].is_border) {
					he = half_edges[halfedge.inv].next;
					continue;
				}
				float3 point_c = vertices[half_edges[halfedge.inv].origin].position;
				float3 point_a = vertices[half_edges[halfedge.prev].origin].position;
				//float3 point_b = vert.position;
				float3 edge_vector_ab = vert.position - point_a;
				float3 edge_vector_bc = point_c - vert.position;
				normal += normalized(cross3df(edge_vector_ab, edge_vector_bc));

				he = half_edges[halfedge.inv].next;
			} while (he != vert.he);
			normals[i] = normalized(normal);
		}
	}

	void cpu_kernel_calculate_ring_centroids_gather(Vertex* vertices, HalfEdge* half_edges, float3* centroids, unsigned vertice_count,int stride, int offset) {
		//calculate centroids
		for (int i = offset; i < vertice_count; i += stride) {
			auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}

			float3 centroid;
			centroid.x = 0.f;
			centroid.y = 0.f;
			centroid.z = 0.f;

			int he = vert.he;
			unsigned neighbors = 0;
			do {//for every neighbor
				HalfEdge& halfedge = half_edges[he];
				HalfEdge& inv_halfedge = half_edges[halfedge.inv];
				float3 p = vertices[inv_halfedge.origin].position;
				centroid += p;
				++neighbors;
				he = inv_halfedge.next;
			} while (he != vert.he);
			centroid.x /= neighbors;
			centroid.y /= neighbors;
			centroid.z /= neighbors;
			centroids[i] = centroid;
		}
	}

	void cpu_kernel_normals_scatter(float3* positions, int* faces, int* face_indices, float3* normals, int face_count,int stride,int offset) {
		for (int i = offset; i < face_count - 1; i += stride) {
			int base_index = faces[i];
			int next_face = faces[i + 1];

			float3 point_a = positions[face_indices[next_face - 1]];
			float3 point_b = positions[face_indices[base_index]];
			float3 point_c = positions[face_indices[base_index + 1]];
			float3 edge_vector_ab = point_b - point_a;
			float3 edge_vector_bc = point_c - point_b;
			float3 normal{ 0.f,0.f,0.f };
			//assume planar polygon
			normal += normalized(cross3df(edge_vector_ab, edge_vector_bc));
			//add to every vertice in the face
			for (int j = 0; j < next_face - base_index; ++j) {
				float3* vn = &normals[face_indices[base_index + j]];
				atomic_add(&vn->x, normal.x);
				atomic_add(&vn->y, normal.y);
				atomic_add(&vn->z, normal.z);
			}
		}
	}

	void cpu_kernel_calculate_ring_centroids_scatter_no_borders(float3* positions, int* faces, int* face_indices,
				int* face_sizes, float3* centroids, int* duped_neighbor_counts, int face_count,int stride,int offset) {
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			//circulate trough the face and add it to the centroids
			for (int j = 0; j < face_size; ++j) {
				float3 next = positions[face_indices[base_index + ((j + 1) % face_size)]];
				//float3 prev = positions[face_indices[base_index + ((j-1) % face_size)]];

				float3* centroid = centroids + face_indices[base_index + j];
				int* neighbor_count = duped_neighbor_counts + face_indices[base_index + j];
				atomic_add(&centroid->x, next.x);
				atomic_add(&centroid->y, next.y);
				atomic_add(&centroid->z, next.z);
				atomic_add(neighbor_count, 1);
			}
		}
	}

	void cpu_kernel_calculate_ring_centroids_scatter(float3* positions, pair<int, int>* edges, float3* centroids,
			int* neighbor_counts, int edge_count,int stride,int offset) {
		for (int i = offset; i < edge_count; i += stride) {
			pair<int, int> edge = edges[i];
			if (edge.first > -1) {
				float3* centroid_a = centroids + edge.first;
				float3* centroid_b = centroids + edge.second;
				float3 pa = positions[edge.first];
				float3 pb = positions[edge.second];
				atomic_add(&centroid_a->x, pb.x);
				atomic_add(&centroid_a->y, pb.y);
				atomic_add(&centroid_a->z, pb.z);
				atomic_add(&centroid_b->x, pa.x);
				atomic_add(&centroid_b->y, pa.y);
				atomic_add(&centroid_b->z, pa.z);
				atomic_add(neighbor_counts + edge.first, 1);
				atomic_add(neighbor_counts + edge.second, 1);
			}
		}
	}
	
	void find_edges(pair<int, int>* pairs, int* faces, int* face_indices, int face_count, int face_index_count) {
		int face = 0; //face of the first vertex in the pair
		int face_start = faces[0];
		int next_face_start = faces[1];
		for (int i = 0; i + 1 < face_index_count; i++) {
			//check current face and next face
			if (next_face_start <= i) {
				++face;
				face_start = faces[face];
				next_face_start = faces[face+1];
			}
			int first, second;
			//check for edge
			if (next_face_start == i + 1) {
				second = face_indices[face_start];
				first = face_indices[i];
			}
			else {
				first = face_indices[i];
				second = face_indices[i + 1];
			}

			if (first > second) {
				pairs[i] = pair<int, int>(second, first);
			}
			else {
				pairs[i] = pair<int, int>(first, second);
			}
		}
	}

	struct PairLessThan {
		bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
			return a.first < b.first || (a.first == b.first && a.second < b.second);
		}
	};

	void normals_he_cpu(HalfedgeMesh* mesh, int threads, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_gather, mesh->vertices.data(), mesh->half_edges.data(),
				mesh->loops.data(), mesh->normals.data(), mesh->vertices.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		timing.processing_time = timing.kernel_execution_time_a;
	}

	void normals_sm_cpu(SimpleMesh* mesh, int threads, timing_struct& timing) {
		mesh->normals.resize(mesh->positions.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		std::vector<std::thread> thread_list;
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_scatter, mesh->positions.data(), mesh->face_starts.data(),
				mesh->faces.data(), mesh->normals.data(),mesh->face_starts.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		thread_list.resize(0);
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normalize_vectors, mesh->normals.data(), mesh->normals.size(), threads, i));
		}
		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}

		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		timing.processing_time = timing.kernel_execution_time_a;
	}

	void centroids_he_cpu(HalfedgeMesh* mesh, attribute_vector<float3>& centroids_array, size_t threads, timing_struct& timing) {
		centroids_array.clear();
		centroids_array.resize(mesh->vertices.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		std::vector<std::thread> thread_list;
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_gather,mesh->vertices.data(), mesh->half_edges.data(), centroids_array.data(),
				mesh->vertices.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		timing.processing_time = timing.kernel_execution_time_a;
	}

	void centroids_sm_cpu(SimpleMesh* mesh, attribute_vector<float3>& centroids_array, size_t threads, timing_struct& timing) {
		centroids_array.clear();
		centroids_array.resize(mesh->positions.size(), { 0,0,0 });
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<int> neighbor_count(mesh->positions.size(), 0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		std::vector<std::thread> thread_list;
		thread_list.reserve(threads);
		attribute_vector<std::pair<int, int>> edges(mesh->faces.size()-1, std::pair<int, int>(-1, -1));//max size == edgecount <= face_indices - 1
		start = std::chrono::steady_clock::now();
		find_edges(edges.data(), mesh->face_starts.data(), mesh->faces.data(), mesh->face_starts.size(),mesh->faces.size());
		std::sort(edges.begin(), edges.end(), PairLessThan());
		auto end = std::unique(edges.begin(), edges.end());
		edges.resize(end-edges.begin());
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_prepare = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		start = std::chrono::steady_clock::now();
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_scatter, mesh->positions.data(), edges.data(),
				centroids_array.data(), neighbor_count.data(),edges.size(),
				threads, i));
		}
		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		thread_list.resize(0);
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_divide,centroids_array.data(), neighbor_count.data(),centroids_array.size(),threads, i));
		}
		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//timing.processing_time = timing.kernel_execution_time_a+timing.kernel_execution_time_prepare;
		timing.processing_time = timing.kernel_execution_time_a;
	}
}