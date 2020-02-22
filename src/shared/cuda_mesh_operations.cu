#include <cuda_mesh_operations.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
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
	cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,kernel, 0, 256);
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

	__global__ void kernel_calculate_normals_gather_area_weight(Vertex* vertices, HalfEdge* half_edges,Loop* loops, float3* normals, unsigned vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		
		//calculate normals
		for (int i = offset; i < vertice_count; i+=stride) {
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
					normal += cross3df(a, b);
					he = halfedge.next;
				} while (he != base_he);
				base_he = half_edges[half_edges[base_he].inv].next;
			} while (base_he != vert.he);
			normals[i] = normalized(normal);
		}
	}

	__global__ void kernel_calculate_ring_centroids_gather(Vertex* vertices, HalfEdge* half_edges, float3* centroids, unsigned vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();

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
	
	__global__ void kernel_calculate_ring_centroids_scatter_no_borders(float3* positions, int* faces, int* face_indices, int* face_sizes, float3* centroids, int* duped_neighbor_counts, int face_count) {
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

	__global__ void kernel_calculate_ring_centroids_scatter(float3* positions, pair<int,int>* edges, float3* centroids, int* neighbor_counts, int edge_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < edge_count; i += stride) {
			pair<int, int> edge = edges[i];
			if (edge.first > -1 && edge.second > -1) {
				float3* centroid_a = centroids + edge.first;
				float3* centroid_b = centroids + edge.second;
				float3 pa = positions[edge.first];
				float3 pb = positions[edge.second];
				atomicAdd(&centroid_a->x, pb.x);
				atomicAdd(&centroid_a->y, pb.y);
				atomicAdd(&centroid_a->z, pb.z);
				atomicAdd(&centroid_b->x, pa.x);
				atomicAdd(&centroid_b->y, pa.y);
				atomicAdd(&centroid_b->z, pa.z);
				atomicAdd(neighbor_counts + edge.first, 1);
				atomicAdd(neighbor_counts + edge.second, 1);
			}
		}
	}

	void find_edges(pair<int,int>* pairs, int* faces, int* face_indices, int* face_sizes,int face_count,int face_index_count) {
		//int stride = thread_stride();
		//int offset = thread_offset();
		int face = 0; //face of the first vertex in the pair
		int face_start = faces[0];
		int next_face_start = faces[0] + face_sizes[0];
		for (int i = 0; i+1 < face_index_count; i++) {
			//check current face and next face
			if (next_face_start <= i) {
				++face;
				face_start = faces[face];
				next_face_start = face_start + face_sizes[face];
			}
			int first, secound;
			//check for edge
			if (next_face_start == i+1) {
				secound = face_indices[face_start];
				first = face_indices[i];
			}
			else {
				first = face_indices[i];
				secound = face_indices[i + 1];
			}

			if (first > secound) {
				pairs[i] = pair<int, int>(secound, first);
			}
			else {
				pairs[i] = pair<int, int>(first, secound);
			}
		}
	}

	struct PairLessThan {
		__device__  __host__ bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
			return a.first < b.first || (a.first == b.first && a.second < b.second);
		}
	};


	void normals_by_area_weight_he_cuda(HalfedgeMesh* mesh, int threads,int blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size()); //prepare vector for normals
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_gather_area_weight);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Loop> loops = mesh->loops;
		thrust::device_vector<float3> normals = mesh->normals;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//setup timer
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//kernel launch
		cudaEventRecord(cu_start);
		kernel_calculate_normals_gather_area_weight<<<blocks,threads>>>(vertices.data().get(), 
				halfedges.data().get(),loops.data().get(), normals.data().get(), vertices.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
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
		kernel_normalize_vectors<<<blocks, threads>>>(normals.data().get(),normals.size());
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
		centroids_array.resize(mesh->vertices.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_gather);
		timing.block_size = threads;
		timing.grid_size = blocks;

		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Loop> loops = mesh->loops;
		thrust::device_vector<float3> centroids = centroids_array;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//create events
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//launch kernel
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_gather <<<blocks, threads >>> (vertices.data().get(), halfedges.data().get(), centroids.data().get(), vertices.size());
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
		
		std::vector<pair<int, int>> edges(faces_indices.size()-1,pair<int,int>(-2,-2));//max size == edgecount <= face_indices - 1
		find_edges(edges.data(), mesh->faces.data(), mesh->face_indices.data(),
				mesh->face_sizes.data(), mesh->faces.size(), mesh->face_indices.size());
		thrust::device_vector<pair<int, int>> dev_edges = edges;
		thrust::sort(dev_edges.begin(), dev_edges.end(), PairLessThan());
		thrust::unique(dev_edges.begin(), dev_edges.end());
		thrust::copy(dev_edges.begin(), dev_edges.end(), edges.begin());
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		cudaEventRecord(cu_start);
		//kernel_calculate_ring_centroids_scatter_no_borders<<<blocks, threads>>>(positions.data().get(), faces.data().get(),
		//		faces_indices.data().get(), faces_sizes.data().get(), centroids.data().get(),neighbor_count.data().get(), faces.size());
		kernel_calculate_ring_centroids_scatter<<<blocks, threads>>>(positions.data().get(),dev_edges.data().get(), centroids.data().get(),neighbor_count.data().get(), dev_edges.size());
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

	//causes the cuda driver to load to prevent loads on mesuring
	void prepare_device() {
		kernel_train<<<1,256>>>();
		cudaDeviceSynchronize();
	}

}



