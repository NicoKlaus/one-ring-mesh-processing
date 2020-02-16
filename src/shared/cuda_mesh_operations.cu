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

	__global__ void kernel_normalize_vectors(float* vec_x, float* vec_y, float* vec_z,unsigned size){
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < size; i += stride) {
			float x = vec_x[i];
			float y = vec_y[i];
			float z = vec_z[i];
			float rnorm = rsqrtf(x * x + y * y + z * z);
			vec_x[i] = rnorm*x;
			vec_y[i] = rnorm*y;
			vec_z[i] = rnorm*z;
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

	__global__ void kernel_divide_vectors(float* vec_x, float* vec_y, float* vec_z, int* div, unsigned vec_size) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < vec_size; i += stride) {
			float fdiv = static_cast<float>(div[i]);
			vec_x[i] /= fdiv;
			vec_y[i] /= fdiv;
			vec_z[i] /= fdiv;
		}
	}

	__global__ void kernel_calculate_normals_scatter_area_weight(
			float* positions_x, float* positions_y, float* positions_z,
			float* normals_x, float* normals_y, float* normals_z, int* faces, int* face_indices, int* face_sizes, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];
			
			int pa_idx = face_indices[base_index + (face_size - 1)];
			int pb_idx = face_indices[base_index];
			float point_b_x = positions_x[pb_idx];
			float point_b_y = positions_y[pb_idx];
			float point_b_z = positions_z[pb_idx];
			float edge_vector_ab_x = point_b_x- positions_x[pa_idx];
			float edge_vector_ab_y = point_b_y- positions_y[pa_idx];
			float edge_vector_ab_z = point_b_z- positions_z[pa_idx];
			float normal_x = 0.f;
			float normal_y = 0.f;
			float normal_z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0;j< face_size;++j){
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
			for (int j = 0;j< face_size;++j){
				int n_idx = face_indices[base_index + j];
				atomicAdd(normals_x+n_idx, normal_x);
				atomicAdd(normals_y+n_idx, normal_y);
				atomicAdd(normals_z+n_idx, normal_z);
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
				float3 normal{ 0,0,0 };
				vertex_nx[i] = normal.x;
				vertex_ny[i] = normal.y;
				vertex_nz[i] = normal.z;
				continue;
			}

			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;

			int base_he = vert_he;
			do {//for every neighbor
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

					normal.x += ay * bz - by * az;
					normal.y += az * bx - bz * ax;
					normal.z += ax * by - bx * ay;
					he = he_next;
				} while (he != base_he);
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
				float* centroids_x, float* centroids_y, float* centroids_z, unsigned vertice_count) {
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
			centroids_x[i] = centroid.x;
			centroids_y[i] = centroid.y;
			centroids_z[i] = centroid.z;
		}
	}
	
	__global__ void kernel_calculate_ring_centroids_scatter(
			float* positions_x, float* positions_y, float* positions_z,
			float* centroids_x, float* centroids_y, float* centroids_z,
			int* faces, int* face_indices, int* face_sizes, int* neighbor_counts, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			//circulate trough the face and add it to the centroids
			for (int j = 0; j < face_size; ++j) {
				int next_idx = face_indices[base_index + ((j + 1) % face_size)];
				int centroid_idx = face_indices[base_index + j];
				atomicAdd(centroids_x+centroid_idx, positions_x[next_idx]);
				atomicAdd(centroids_y+centroid_idx, positions_y[next_idx]);
				atomicAdd(centroids_z+centroid_idx, positions_z[next_idx]);
				atomicAdd(neighbor_counts + centroid_idx, 1);
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
		mesh->clear_normals();
		mesh->resize_normals(mesh->vertex_count());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_scatter_area_weight);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<float> positions_x = mesh->positions_x;
		thrust::device_vector<float> positions_y = mesh->positions_y;
		thrust::device_vector<float> positions_z = mesh->positions_z;
		thrust::device_vector<float> normals_x(mesh->vertex_count());
		thrust::device_vector<float> normals_y(mesh->vertex_count());
		thrust::device_vector<float> normals_z(mesh->vertex_count());
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		cudaEventRecord(cu_start);
		kernel_calculate_normals_scatter_area_weight<<<blocks, threads>>>(positions_x.data().get(), positions_y.data().get(), positions_z.data().get(),
				normals_x.data().get(), normals_y.data().get(), normals_z.data().get(),
				faces.data().get(), faces_indices.data().get(), faces_sizes.data().get(),faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//run secound kernel
		cudaEventRecord(cu_start);
		kernel_normalize_vectors<<<blocks, threads>>>(normals_x.data().get(), normals_y.data().get(), normals_z.data().get(),normals_x.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);

		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();
		thrust::copy(normals_x.begin(), normals_x.end(), mesh->normals_x.begin());
		thrust::copy(normals_y.begin(), normals_y.end(), mesh->normals_y.begin());
		thrust::copy(normals_z.begin(), normals_z.end(), mesh->normals_z.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cuda(HalfedgeMesh* mesh,
			std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z, 
			int threads,int blocks, timing_struct& timing) {
		centroids_array_x.resize(mesh->vertex_count());
		centroids_array_y.resize(mesh->vertex_count());
		centroids_array_z.resize(mesh->vertex_count());
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
		thrust::device_vector<float> centroids_x = centroids_array_x;
		thrust::device_vector<float> centroids_y = centroids_array_y;
		thrust::device_vector<float> centroids_z = centroids_array_z;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//create events
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//launch kernel
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_gather<<<blocks, threads>>>(vertex_x.data().get(), vertex_y.data().get(), vertex_z.data().get(),
			vertex_he.data().get(),halfedge_next.data().get(), halfedge_inv.data().get(), halfedge_origins.data().get(),
			centroids_x.data().get(), centroids_y.data().get(), centroids_z.data().get(), mesh->vertex_count());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		//read back
		start = std::chrono::steady_clock::now();
		thrust::copy(centroids_x.begin(), centroids_x.end(), centroids_array_x.begin());
		thrust::copy(centroids_y.begin(), centroids_y.end(), centroids_array_y.begin());
		thrust::copy(centroids_z.begin(), centroids_z.end(), centroids_array_z.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cuda(SimpleMesh* mesh,
			std::vector<float>& centroids_array_x, std::vector<float>& centroids_array_y, std::vector<float>& centroids_array_z,
			int threads,int blocks, timing_struct& timing) {
		centroids_array_x.resize(mesh->vertex_count());
		centroids_array_y.resize(mesh->vertex_count());
		centroids_array_z.resize(mesh->vertex_count());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_scatter);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<float> positions_x = mesh->positions_x;
		thrust::device_vector<float> positions_y = mesh->positions_y;
		thrust::device_vector<float> positions_z = mesh->positions_z;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float> centroids_x = centroids_array_x;
		thrust::device_vector<float> centroids_y = centroids_array_y;
		thrust::device_vector<float> centroids_z = centroids_array_z;
		thrust::device_vector<int> neighbor_count(mesh->vertex_count(),0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_scatter<<<blocks, threads>>>(positions_x.data().get(), positions_y.data().get(), positions_z.data().get(),
			centroids_x.data().get(), centroids_y.data().get(), centroids_z.data().get(),
			faces.data().get(), faces_indices.data().get(), faces_sizes.data().get(), neighbor_count.data().get(), faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//divide
		cudaEventRecord(cu_start);
		kernel_divide_vectors<<<blocks, threads>>>(centroids_x.data().get(), centroids_y.data().get(), centroids_z.data().get(),
				neighbor_count.data().get(), centroids_x.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);

		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();
		thrust::copy(centroids_x.begin(), centroids_x.end(), centroids_array_x.begin());
		thrust::copy(centroids_y.begin(), centroids_y.end(), centroids_array_y.begin());
		thrust::copy(centroids_z.begin(), centroids_z.end(), centroids_array_z.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

}



