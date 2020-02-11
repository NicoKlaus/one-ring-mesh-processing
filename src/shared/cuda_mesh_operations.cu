#include <cuda_mesh_operations.hpp>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <timing_struct.hpp>

using namespace thrust;

namespace ab {

__device__ int thread_offset(){
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int thread_stride(){
	return blockDim.x * gridDim.x;
}

template <typename T>
__host__ void optimal_configuration(int& blocks, int& threads,const T& kernel) {
	int grid_size = 0;
	int block_size = 0;
	cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,kernel, 0, 0);
	blocks = grid_size;
	threads = block_size;
}

__host__ size_t cuda_elapsed_time(cudaEvent_t& cu_start, cudaEvent_t& cu_stop) {
	float exec_time;
	cudaEventElapsedTime(&exec_time, cu_start, cu_stop);
	return static_cast<size_t>(exec_time * 1000000);
}

#if __CUDA_ARCH__ < 600
__device__ int atomicAdd(int* address, int val)
{
	int old = *address, assumed;

	do {
		assumed = old;
		old = atomicCAS(address, assumed,val + assumed);
	} while (assumed != old);

	return old;
}
#endif

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#if __CUDA_ARCH__ < 600
__device__ float atomicAdd(float* address, float val)
{
	unsigned int* address_as_ull =
		(unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__float_as_int(val +
				__int_as_float(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __int_as_float(old);
}
#endif

	__global__ void kernel_train() {
		
	}

	__global__ void kernel_normalize_vectors(float3* vec,unsigned size){
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < size; i += stride) {
			vec[i] = normalized(vec[i]);
		}
	}
	
	__global__ void kernel_divide(float3* vec,float* div,unsigned vec_size){
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < vec_size; i += stride) {
			float fdiv = div[i];
			vec[i].x /= fdiv;
			vec[i].y /= fdiv;
			vec[i].z /= fdiv;
		}
	}

	__global__ void kernel_divide(float3* vec, int* div, unsigned vec_size) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < vec_size; i += stride) {
			float fdiv = static_cast<float>(div[i]);
			vec[i].x /= fdiv;
			vec[i].y /= fdiv;
			vec[i].z /= fdiv;
		}
	}

	__global__ void kernel_calculate_normals_scatter_area_weight(float3* positions,int* faces,int* face_indices,int* face_sizes, float3* normals, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];
			
			float3 point_a = positions[face_indices[base_index+(face_size-1)]];
			float3 point_b = positions[face_indices[base_index]];
			float3 edge_vector_ab = point_b-point_a;
			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0;j< face_size;++j){
				float3 point_c = positions[face_indices[base_index+((j+1)%face_size)]];
				float3 edge_vector_bc = point_c - point_b;
				//adding to the normal vector
				normal += cross3df(edge_vector_ab,edge_vector_bc);
				edge_vector_ab = edge_vector_bc;
			}
			//add to every vertice in the face
			for (int j = 0;j< face_size;++j){
				float3* vn = &normals[face_indices[base_index+j]];
				atomicAdd(&vn->x, normal.x);
				atomicAdd(&vn->y, normal.y);
				atomicAdd(&vn->z, normal.z);
			}
		}
	}

	__global__ void kernel_calculate_normals_gather_area_weight(
				float* vertex_x, float* vertex_y, float* vertex_z,
				float* vertex_nx, float* vertex_ny, float* vertex_nz, int* vertex_he,
				int* halfedge_loops,int* halfedge_origins,int* halfedge_next,int* halfedge_inv,
				bool* loops_is_border, int vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		
		//calculate normals
		for (int i = offset; i < vertice_count; i+=stride) {
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

	__global__ void kernel_calculate_normals_gather_area_weight_no_stride(
		float* vertex_x, float* vertex_y, float* vertex_z,
		float* vertex_nx, float* vertex_ny, float* vertex_nz, int* vertex_he,
		int* halfedge_loops, int* halfedge_origins, int* halfedge_next, int* halfedge_inv,
		bool* loops_is_border, int vertice_count) {
		int i = thread_offset();
		if (i > vertice_count) return;
		//calculate normals
			int vert_he = vertex_he[i];
			if (vert_he == -1) {
				return;
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

	__global__ void kernel_calculate_ring_centroids_gather(
				float* vertex_x, float* vertex_y, float* vertex_z,int* vertex_he,
				int* halfedge_next,int* halfedge_inv,int* halfedge_origins,
				float3* centroids, unsigned vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();

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
			centroids[i] = centroid;
		}
	}
	
	__global__ void kernel_calculate_ring_centroids_scatter(float3* positions, int* faces, int* face_indices, int* face_sizes, float3* centroids, int* duped_neighbor_counts, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			//circulate trough the face and add it to the centroids
			for (int j = 0; j < face_size; ++j) {
				float3 next = positions[face_indices[base_index + ((j+1) % face_size)]];
				//float3 prev = positions[face_indices[base_index + ((j-1) % face_size)]];
				
				float3* centroid = centroids+face_indices[base_index+j];
				int* neighbor_count = duped_neighbor_counts+face_indices[base_index+j];
				atomicAdd(&centroid->x, next.x);
				atomicAdd(&centroid->y, next.y);
				atomicAdd(&centroid->z, next.z);
				atomicAdd(neighbor_count, 1);
			}
		}
	}

	void normals_by_area_weight_he_cuda_no_stride(HalfedgeMesh* mesh, timing_struct& timing) {
		mesh->clear_normals();
		mesh->resize_normals(mesh->vertex_count()); //prepare vector for normals
		

		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<int> halfedge_inv = mesh->half_edge_inv;
		thrust::device_vector<int> halfedge_next = mesh->half_edge_next;
		thrust::device_vector<int> halfedge_origins = mesh->half_edge_origins;
		thrust::device_vector<int> halfedge_loops = mesh->half_edge_loops;
		thrust::device_vector<float> vertex_x = mesh->vertex_positions_x;
		thrust::device_vector<float> vertex_y = mesh->vertex_positions_y;
		thrust::device_vector<float> vertex_z = mesh->vertex_positions_z;
		thrust::device_vector<int> vertex_he = mesh->vertex_he;
		thrust::device_vector<float> vertex_nx(mesh->vertex_count());
		thrust::device_vector<float> vertex_ny(mesh->vertex_count());
		thrust::device_vector<float> vertex_nz(mesh->vertex_count());
		thrust::device_vector<bool> loops_is_border = mesh->loops_is_border;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//setup timer
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//kernel launch
		cudaEventRecord(cu_start);
		int threads = 256;
		int blocks = (mesh->vertex_count() / threads)+ mesh->vertex_count() % threads ? 1 : 0;
		kernel_calculate_normals_gather_area_weight << <blocks, threads >> > (
			vertex_x.data().get(), vertex_y.data().get(), vertex_z.data().get(),
			vertex_nx.data().get(), vertex_ny.data().get(), vertex_nz.data().get(), vertex_he.data().get(),
			halfedge_loops.data().get(), halfedge_origins.data().get(), halfedge_next.data().get(), halfedge_inv.data().get(),
			loops_is_border.data().get(), mesh->vertex_count());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(vertex_nx.begin(), vertex_nx.end(), mesh->normals_x.begin());
		thrust::copy(vertex_ny.begin(), vertex_ny.end(), mesh->normals_y.begin());
		thrust::copy(vertex_nz.begin(), vertex_nz.end(), mesh->normals_z.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		timing.block_size = threads;
		timing.grid_size = blocks;
	}

	void normals_by_area_weight_he_cuda(HalfedgeMesh* mesh, int threads,int blocks, timing_struct& timing) {
		mesh->clear_normals();
		mesh->resize_normals(mesh->vertex_count()); //prepare vector for normals
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_gather_area_weight);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<int> halfedge_inv = mesh->half_edge_inv;
		thrust::device_vector<int> halfedge_next = mesh->half_edge_next;
		thrust::device_vector<int> halfedge_origins = mesh->half_edge_origins;
		thrust::device_vector<int> halfedge_loops = mesh->half_edge_loops;
		thrust::device_vector<float> vertex_x = mesh->vertex_positions_x;
		thrust::device_vector<float> vertex_y = mesh->vertex_positions_y;
		thrust::device_vector<float> vertex_z = mesh->vertex_positions_z;
		thrust::device_vector<int> vertex_he = mesh->vertex_he;
		thrust::device_vector<float> vertex_nx(mesh->vertex_count());
		thrust::device_vector<float> vertex_ny(mesh->vertex_count());
		thrust::device_vector<float> vertex_nz(mesh->vertex_count());
		thrust::device_vector<bool> loops_is_border = mesh->loops_is_border;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//setup timer
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//kernel launch
		cudaEventRecord(cu_start);

		kernel_calculate_normals_gather_area_weight<<<blocks, threads>>>(
			vertex_x.data().get(), vertex_y.data().get(), vertex_z.data().get(),
			vertex_nx.data().get(), vertex_ny.data().get(), vertex_nz.data().get(), vertex_he.data().get(),
			halfedge_loops.data().get(), halfedge_origins.data().get(), halfedge_next.data().get(), halfedge_inv.data().get(),
			loops_is_border.data().get(), mesh->vertex_count());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(vertex_nx.begin(), vertex_nx.end(), mesh->normals_x.begin());
		thrust::copy(vertex_ny.begin(), vertex_ny.end(), mesh->normals_y.begin());
		thrust::copy(vertex_nz.begin(), vertex_nz.end(), mesh->normals_z.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	/// normals from a simple mesh
	void normals_by_area_weight_sm_cuda(SimpleMesh* mesh,int threads,int blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->positions.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_scatter_area_weight);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<float3> positions = mesh->positions;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float3> normals = mesh->normals;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		cudaEventRecord(cu_start);
		kernel_calculate_normals_scatter_area_weight<<<blocks, threads>>>(positions.data().get(), faces.data().get(), faces_indices.data().get(), faces_sizes.data().get(), normals.data().get(), faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//run secound kernel
		cudaEventRecord(cu_start);
		kernel_normalize_vectors<<<1, threads>>>(normals.data().get(),normals.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);

		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cuda(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, int threads,int blocks, timing_struct& timing) {
		centroids_array.resize(mesh->vertex_count());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_gather);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<float> vertex_x = mesh->vertex_positions_x;
		thrust::device_vector<float> vertex_y = mesh->vertex_positions_y;
		thrust::device_vector<float> vertex_z = mesh->vertex_positions_z;
		thrust::device_vector<int> vertex_he = mesh->vertex_he;
		thrust::device_vector<int> halfedge_inv = mesh->half_edge_inv;
		thrust::device_vector<int> halfedge_next = mesh->half_edge_next;
		thrust::device_vector<int> halfedge_origins = mesh->half_edge_origins;
		thrust::device_vector<float3> centroids = centroids_array;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//create events
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//launch kernel
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_gather<<<blocks, threads>>>(vertex_x.data().get(), vertex_y.data().get(), vertex_z.data().get(), vertex_he.data().get(),
			halfedge_next.data().get(), halfedge_inv.data().get(), halfedge_origins.data().get(), centroids.data().get(), mesh->vertex_count());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		//read back
		start = std::chrono::steady_clock::now();
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cuda(SimpleMesh* mesh, std::vector<float3>& centroids_array, int threads,int blocks, timing_struct& timing) {
		centroids_array.resize(mesh->positions.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_scatter);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<float3> positions = mesh->positions;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float3> centroids = centroids_array;
		thrust::device_vector<int> neighbor_count(mesh->positions.size(),0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_scatter<<<blocks, threads>>>(positions.data().get(), faces.data().get(),
				faces_indices.data().get(), faces_sizes.data().get(), centroids.data().get(),neighbor_count.data().get(), faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//divide
		cudaEventRecord(cu_start);
		kernel_divide<<<blocks, threads>>>(centroids.data().get(), neighbor_count.data().get(), centroids.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);

		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

}



