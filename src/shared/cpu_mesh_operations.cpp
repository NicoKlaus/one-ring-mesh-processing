#include <cpu_mesh_operations.hpp>
#include <intrin.h>

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

	void cpu_kernel_normals_by_are_weight_gather(Vertex* vertices, HalfEdge* half_edges,
			Loop* loops, float3* normals, unsigned vertice_count, int stride, int offset) {

		//calculate normals
		for (int i = offset; i < vertice_count; i += stride) {
			auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}

			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;

			int base_he = vert.he;
			do {//for every neighbor
				float3 pnormal;
				pnormal.x = 0.f;
				pnormal.y = 0.f;
				pnormal.z = 0.f;
				int he = base_he;
				//skip boundary loops
				if (loops[half_edges[base_he].loop].is_border) {
					base_he = half_edges[half_edges[base_he].inv].next;
					continue;
				}
				do {//calculate polygon normal
					HalfEdge& halfedge = half_edges[he];
					float3 a = vertices[halfedge.origin].position;
					float3 b = vertices[half_edges[halfedge.next].origin].position;
					pnormal = pnormal + cross3df(a, b);
					he = halfedge.next;
				} while (he != base_he);
				normal += pnormal;
				base_he = half_edges[half_edges[base_he].inv].next;
			} while (base_he != vert.he);
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

	void cpu_kernel_normals_by_area_weight_scatter(float3* positions, int* faces, int* face_indices,
			int* face_sizes, float3* normals, int face_count, int stride, int offset) {
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			float3 point_a = positions[face_indices[base_index + (face_size - 1)]];
			float3 point_b = positions[face_indices[base_index]];
			float3 edge_vector_ab = point_b - point_a;
			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0; j < face_size; ++j) {
				float3 point_c = positions[face_indices[base_index + ((j + 1) % face_size)]];
				float3 edge_vector_bc = point_c - point_b;
				//adding to the normal vector
				normal += cross3df(edge_vector_ab, edge_vector_bc);
				edge_vector_ab = edge_vector_bc;
			}
			//add to every vertice in the face
			for (int j = 0; j < face_size; ++j) {
				float3* vn = &normals[face_indices[base_index + j]];
				atomic_add(&vn->x, normal.x);
				atomic_add(&vn->y, normal.y);
				atomic_add(&vn->z, normal.z);
			}
		}
	}

	__global__ void cpu_kernel_calculate_ring_centroids_scatter(float3* positions, int* faces, int* face_indices,
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

	void normals_by_area_weight_he_cpu(HalfedgeMesh* mesh, int threads, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_by_are_weight_gather, mesh->vertices.data(), mesh->half_edges.data(),
				mesh->loops.data(), mesh->normals.data(), mesh->vertices.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void normals_by_area_weight_sm_cpu(SimpleMesh* mesh, int threads, timing_struct& timing) {
		mesh->normals.resize(mesh->positions.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_by_area_weight_scatter, mesh->positions.data(), mesh->faces.data(),
				mesh->face_indices.data(), mesh->face_sizes.data(), mesh->normals.data(),mesh->faces.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cpu(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, size_t threads, timing_struct& timing) {
		centroids_array.resize(mesh->vertices.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_gather,mesh->vertices.data(), mesh->half_edges.data(), centroids_array.data(),
				mesh->vertices.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cpu(SimpleMesh* mesh, std::vector<float3>& centroids_array, size_t threads, timing_struct& timing) {
		centroids_array.resize(mesh->positions.size());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<int> neighbor_count(mesh->positions.size(), 0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_scatter, mesh->positions.data(), mesh->faces.data(),
				mesh->face_indices.data(), mesh->face_sizes.data(), centroids_array.data(), neighbor_count.data(), mesh->faces.size(),
				threads, i));;
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}

		for (int i = 0; i < centroids_array.size(); ++i) {
			centroids_array[i].x /= neighbor_count[i];
			centroids_array[i].y /= neighbor_count[i];
			centroids_array[i].z /= neighbor_count[i];
		}

		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}
}