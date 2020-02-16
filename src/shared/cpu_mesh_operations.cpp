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
			old = _InterlockedCompareExchange(address_as_ull,
				assumed, reinterpret_cast<long&>(new_value));

			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
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
			old = _InterlockedCompareExchange(address_as_ull,
				assumed, reinterpret_cast<long&>(new_value));

			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);

		return  reinterpret_cast<int&>(old);
	}


	void cpu_kernel_normals_by_are_weight_gather(
		float* vertex_x, float* vertex_y, float* vertex_z,
		float* vertex_nx, float* vertex_ny, float* vertex_nz, int* vertex_he,
		int* halfedge_loops, int* halfedge_origins, int* halfedge_next, int* halfedge_inv,
		bool* loops_is_border, int vertice_count,int stride,int offset) {

		//calculate normals
		for (int i = offset; i < vertice_count; i += stride) {
			int vert_he = vertex_he[i];
			if (vert_he == -1) {
				continue;
			}

			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;

			int base_he = vert_he;
			do {//for every neighbor
				float3 pnormal;
				pnormal.x = 0.f;
				pnormal.y = 0.f;
				pnormal.z = 0.f;
				int he = base_he;
				//skip boundary loops
				if (loops_is_border[halfedge_loops[base_he]]) {
					base_he = halfedge_next[halfedge_inv[base_he]];
					continue;
				}
				do {//calculate polygon normal
					int he_origin = halfedge_origins[he];
					int he_next = halfedge_next[he];
					int he_next_origin = halfedge_origins[he_next];
					float ax, ay, az, bx, by, bz;
					ax = vertex_x[he_origin];
					ay = vertex_y[he_origin];
					az = vertex_z[he_origin];
					bx = vertex_x[he_next_origin];
					by = vertex_y[he_next_origin];
					bz = vertex_z[he_next_origin];

					pnormal.x += ay * bz - by * az;
					pnormal.y += az * bx - bz * ax;
					pnormal.z += ax * by - bx * ay;
					he = he_next;
				} while (he != base_he);
				normal += pnormal;
				base_he = halfedge_next[halfedge_inv[base_he]];
			} while (base_he != vert_he);
			//normalize
			normal = normalized(normal);
			vertex_nx[i] = normal.x;
			vertex_ny[i] = normal.y;
			vertex_nz[i] = normal.z;
		}
	}

	
	void cpu_kernel_calculate_ring_centroids_gather(
		float* vertex_x, float* vertex_y, float* vertex_z, int* vertex_he,
		int* halfedge_next, int* halfedge_inv, int* halfedge_origins,
		float* centroids_x,float* centroids_y,float* centroids_z, unsigned vertice_count,int stride,int offset) {

		//calculate centroids
		for (int i = offset; i < vertice_count; i += stride) {
			int vert_he = vertex_he[i];
			if (vert_he == -1) {
				continue;
			}

			float3 centroid;
			centroid.x = 0.f;
			centroid.y = 0.f;
			centroid.z = 0.f;

			int he = vert_he;
			int neighbors = 0;
			do {//for every neighbor
				int he_inv = halfedge_inv[he];
				int he_inv_origin = halfedge_origins[he_inv];
				float3 p;
				p.x = vertex_x[he_inv_origin];
				p.y = vertex_y[he_inv_origin];
				p.z = vertex_z[he_inv_origin];
				centroid += p;
				++neighbors;
				he = halfedge_next[he_inv];
			} while (he != vert_he);
			centroid.x /= neighbors;
			centroid.y /= neighbors;
			centroid.z /= neighbors;
			centroids_x[i] = centroid.x;
			centroids_y[i] = centroid.y;
			centroids_z[i] = centroid.z;
		}
	}

	void cpu_kernel_normals_by_area_weight_scatter(
		float* positions_x, float* positions_y, float* positions_z,
		float* normals_x, float* normals_y, float* normals_z, int* faces, int* face_indices, int* face_sizes, int face_count,int stride,int offset) {
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			int pa_idx = face_indices[base_index + (face_size - 1)];
			int pb_idx = face_indices[base_index];
			float point_b_x = positions_x[pb_idx];
			float point_b_y = positions_y[pb_idx];
			float point_b_z = positions_z[pb_idx];
			float edge_vector_ab_x = point_b_x - positions_x[pa_idx];
			float edge_vector_ab_y = point_b_y - positions_y[pa_idx];
			float edge_vector_ab_z = point_b_z - positions_z[pa_idx];
			float normal_x = 0.f;
			float normal_y = 0.f;
			float normal_z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0; j < face_size; ++j) {
				int pc_idx = face_indices[base_index + ((j + 1) % face_size)];
				float edge_vector_bc_x = positions_x[pc_idx] - point_b_x;
				float edge_vector_bc_y = positions_y[pc_idx] - point_b_y;
				float edge_vector_bc_z = positions_z[pc_idx] - point_b_z;
				//adding to the normal vector
				normal_x += edge_vector_ab_y * edge_vector_bc_z - edge_vector_bc_y * edge_vector_ab_z;
				normal_y += edge_vector_ab_z * edge_vector_bc_x - edge_vector_bc_z * edge_vector_ab_x;
				normal_z += edge_vector_ab_x * edge_vector_bc_y - edge_vector_bc_x * edge_vector_ab_y;
				edge_vector_ab_x = edge_vector_bc_x;
				edge_vector_ab_y = edge_vector_bc_y;
				edge_vector_ab_z = edge_vector_bc_z;
			}
			//add to every vertice in the face
			for (int j = 0; j < face_size; ++j) {
				int n_idx = face_indices[base_index + j];
				atomic_add(normals_x + n_idx, normal_x);
				atomic_add(normals_y + n_idx, normal_y);
				atomic_add(normals_z + n_idx, normal_z);
			}
		}
	}

	void cpu_kernel_calculate_ring_centroids_scatter(
		float* positions_x, float* positions_y, float* positions_z,
		float* centroids_x, float* centroids_y, float* centroids_z,
		int* faces, int* face_indices, int* face_sizes, int* neighbor_counts, int face_count,int stride,int offset) {
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			//circulate trough the face and add it to the centroids
			for (int j = 0; j < face_size; ++j) {
				int next_idx = face_indices[base_index + ((j + 1) % face_size)];
				int centroid_idx = face_indices[base_index + j];
				atomic_add(centroids_x + centroid_idx, positions_x[next_idx]);
				atomic_add(centroids_y + centroid_idx, positions_y[next_idx]);
				atomic_add(centroids_z + centroid_idx, positions_z[next_idx]);
				atomic_add(neighbor_counts + centroid_idx, 1);
			}
		}
	}


	void normals_by_area_weight_he_cpu(HalfedgeMesh* mesh, int threads, timing_struct& timing) {
		mesh->clear_normals();
		mesh->resize_normals(mesh->vertex_count());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_by_are_weight_gather, 
				mesh->vertex_positions_x.data(), mesh->vertex_positions_y.data(), mesh->vertex_positions_z.data(),
				mesh->normals_x.data(), mesh->normals_y.data(), mesh->normals_z.data(),mesh->vertex_he.data(),
				mesh->half_edge_loops.data(),mesh->half_edge_origins.data(),mesh->half_edge_next.data(),mesh->half_edge_inv.data(),
				reinterpret_cast<bool*>(mesh->loops_is_border.data()),mesh->vertex_count(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void normals_by_area_weight_sm_cpu(SimpleMesh* mesh, int threads, timing_struct& timing) {
		mesh->resize_normals(mesh->vertex_count());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_normals_by_area_weight_scatter, 
				mesh->positions_x.data(), mesh->positions_y.data(), mesh->positions_z.data(), 
				mesh->normals_x.data(), mesh->normals_y.data(), mesh->normals_z.data(), mesh->faces.data(),
				mesh->face_indices.data(), mesh->face_sizes.data(),mesh->faces.size(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cpu(HalfedgeMesh* mesh, 
			std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z, 
			size_t threads, timing_struct& timing) {
		centroids_array_x.resize(mesh->vertex_count());
		centroids_array_y.resize(mesh->vertex_count());
		centroids_array_z.resize(mesh->vertex_count());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_gather,
				mesh->vertex_positions_x.data(), mesh->vertex_positions_y.data(), mesh->vertex_positions_z.data(),
				mesh->vertex_he.data(),mesh->half_edge_next.data(),mesh->half_edge_inv.data(),mesh->half_edge_origins.data(),
				centroids_array_x.data(), centroids_array_y.data(), centroids_array_z.data(),
				mesh->vertex_count(), threads, i));
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}
		
		auto stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cpu(SimpleMesh* mesh,
			std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z,
			size_t threads, timing_struct& timing) {
		centroids_array_x.resize(mesh->vertex_count());
		centroids_array_y.resize(mesh->vertex_count());
		centroids_array_z.resize(mesh->vertex_count());
		timing.block_size = threads;
		timing.grid_size = 1;

		auto start = std::chrono::steady_clock::now();
		std::vector<int> neighbor_count(mesh->vertex_count(), 0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		start = std::chrono::steady_clock::now();
		std::vector<std::thread> thread_list;
		for (int i = 0; i < threads; ++i) {
			thread_list.emplace_back(std::thread(cpu_kernel_calculate_ring_centroids_scatter,
				mesh->positions_x.data(), mesh->positions_y.data(), mesh->positions_z.data(),
				centroids_array_x.data(), centroids_array_y.data(), centroids_array_z.data(),
				mesh->faces.data(),mesh->face_indices.data(), mesh->face_sizes.data(), neighbor_count.data(), mesh->faces.size(),
				threads, i));;
		}

		for (int i = 0; i < threads; ++i) {
			thread_list[i].join();
		}

		for (int i = 0; i < centroids_array_x.size(); ++i) {
			centroids_array_x[i] /= neighbor_count[i];
			centroids_array_y[i] /= neighbor_count[i];
			centroids_array_z[i] /= neighbor_count[i];
		}

		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}
}