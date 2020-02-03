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

	__global__ void kernel_calculate_face_centroids_scatter(float3* positions, int* faces, int* face_indices, int* face_sizes, float3* centroids, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];

			float3 centroid;
			centroid.x = 0.f;
			centroid.y = 0.f;
			centroid.z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0; j < face_size; ++j) {
				float3 point = positions[face_indices[base_index + j]];
				//adding to the centroid vector
				centroid += point;
			}
			centroid.x /= face_size;
			centroid.y /= face_size;
			centroid.z /= face_size;
			centroids[i] = centroid;
		}
	}

	__global__ void kernel_calculate_face_centroids_gather(Vertex* vertices, HalfEdge* half_edges, Loop* loops, float3* centroids, unsigned loop_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < loop_count; i += stride) {
			auto& loop = loops[i];
			if (loop.is_border) {
				continue;
			}
			int he = loop.he;
			float3 centroid;
			int edge_count = 0;
			centroid.x = 0.f;
			centroid.y = 0.f;
			centroid.z = 0.f;
			do {
				HalfEdge& halfedge = half_edges[he];
				float3 a = vertices[halfedge.origin].position;
				centroid = centroid + a;
				++edge_count;
				he = halfedge.next;
			} while (he != loop.he);
			centroid.x /= (float)edge_count;
			centroid.y /= (float)edge_count;
			centroid.z /= (float)edge_count;
			centroids[i] = centroid;
		}
	}

	void normals_by_area_weight_he_cuda(HalfedgeMesh* mesh, size_t threads,size_t blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size()); //prepare vector for normals
		
		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Loop> loops = mesh->loops;
		thrust::device_vector<float3> normals = mesh->normals;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		start = std::chrono::steady_clock::now(); //kernel launch + execution to synchronisation
		kernel_calculate_normals_gather_area_weight<<<blocks,threads>>>(vertices.data().get(), 
				halfedges.data().get(),loops.data().get(), normals.data().get(), vertices.size());
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	/// normals from a simple mesh
	void normals_by_area_weight_sm_cuda(SimpleMesh* mesh,size_t threads,size_t blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->positions.size());
		
		auto start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<float3> positions = mesh->positions;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float3> normals = mesh->normals;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		start = std::chrono::steady_clock::now();
		kernel_calculate_normals_scatter_area_weight<<<blocks, threads>>>(positions.data().get(), faces.data().get(), faces_indices.data().get(), faces_sizes.data().get(), normals.data().get(), faces.size());
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		start = std::chrono::steady_clock::now();
		kernel_normalize_vectors<<<1, threads>>>(normals.data().get(),normals.size());
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_b = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		start = std::chrono::steady_clock::now();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cuda(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, size_t threads,size_t blocks, timing_struct& timing) {
		centroids_array.resize(mesh->vertices.size());
		
		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Loop> loops = mesh->loops;
		thrust::device_vector<float3> centroids = centroids_array;
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	
		start = std::chrono::steady_clock::now();
		kernel_calculate_ring_centroids_gather <<<blocks, threads >>> (vertices.data().get(), halfedges.data().get(), centroids.data().get(), vertices.size());
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		
		start = std::chrono::steady_clock::now();
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cuda(SimpleMesh* mesh, std::vector<float3>& centroids_array, size_t threads,size_t blocks, timing_struct& timing) {
		centroids_array.resize(mesh->positions.size());
		
		auto start = std::chrono::steady_clock::now();
		thrust::device_vector<float3> positions = mesh->positions;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float3> centroids = centroids_array;
		thrust::device_vector<int> neighbor_count(mesh->positions.size(),0);
		auto stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		start = std::chrono::steady_clock::now();
		kernel_calculate_ring_centroids_scatter<<<blocks, threads>>>(positions.data().get(), faces.data().get(),
				faces_indices.data().get(), faces_sizes.data().get(), centroids.data().get(),neighbor_count.data().get(), faces.size());
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		start = std::chrono::steady_clock::now();
		kernel_divide<<<blocks, threads>>>(centroids.data().get(), neighbor_count.data().get(), centroids.size());
		cudaDeviceSynchronize();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		stop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_b = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		start = std::chrono::steady_clock::now();
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	//face centroids

	void calculate_face_centroids_he_parallel(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, size_t threads, size_t blocks) {
		centroids_array.resize(mesh->loops.size());
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<Loop> loops = mesh->loops;
		thrust::device_vector<float3> centroids(mesh->loops.size(), float3{ 0.f,0.f,0.f });
		kernel_calculate_face_centroids_gather<<<blocks, threads>>>(vertices.data().get(), halfedges.data().get(),
					loops.data().get(), centroids.data().get(), loops.size());
		cudaDeviceSynchronize();
		printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
	}

	void calculate_face_centroids_sm_parallel(SimpleMesh* mesh, std::vector<float3>& centroids_array, size_t threads, size_t blocks) {
		centroids_array.resize(mesh->faces.size());
		thrust::device_vector<float3> positions = mesh->positions;
		thrust::device_vector<int> faces = mesh->faces;
		thrust::device_vector<int> faces_indices = mesh->face_indices;
		thrust::device_vector<int> faces_sizes = mesh->face_sizes;
		thrust::device_vector<float3> centroids(mesh->faces.size(), float3{ 0.f,0.f,0.f });
		kernel_calculate_face_centroids_scatter<<<blocks, threads>>>(positions.data().get(), faces.data().get(),
			faces_indices.data().get(), faces_sizes.data().get(), centroids.data().get(), faces.size());
		cudaDeviceSynchronize();
		printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
	}

}



