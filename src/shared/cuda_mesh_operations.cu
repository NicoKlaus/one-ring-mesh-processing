#include <cuda_mesh_operations.hpp>
#include <cuda_util.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <timing_struct.hpp>
#include <iomanip>

using namespace thrust;

namespace ab {

__device__ int thread_offset(){
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int thread_stride(){
	return blockDim.x * gridDim.x;
}
	__global__ void cuda_sqrtf(float* a) {
		*a = sqrtf(*a);
	}

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
			float fdiv = 1.f/div[i];
			vec[i].x *= fdiv;
			vec[i].y *= fdiv;
			vec[i].z *= fdiv;
		}
	}

	__global__ void kernel_divide(float3* vec, int* div, unsigned vec_size) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < vec_size; i += stride) {
			float fdiv = 1.f/static_cast<float>(div[i]);
			vec[i].x *= fdiv;
			vec[i].y *= fdiv;
			vec[i].z *= fdiv;
		}
	}

	__global__ void kernel_calculate_normals_scatter(float3* positions,int* faces,int* face_indices, float3* normals, int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < face_count-1; i += stride) {
			int base_index = faces[i];
			int next_face = faces[i+1];
			
			float3 point_a = positions[face_indices[next_face-1]];
			float3 point_b = positions[face_indices[base_index]];
			float3 point_c = positions[face_indices[base_index+1]];
			float3 edge_vector_ab = point_b-point_a;
			float3 edge_vector_bc = point_c-point_b;
			float3 normal{ 0.f,0.f,0.f };
			//assume planar polygon
			normal += normalized(cross3df(edge_vector_ab, edge_vector_bc));
			//add to every vertice in the face
			for (int j = 0;j< next_face-base_index;++j){
				float3* vn = &normals[face_indices[base_index+j]];
				atomicAdd(&vn->x, normal.x);
				atomicAdd(&vn->y, normal.y);
				atomicAdd(&vn->z, normal.z);
			}
		}
	}
	//does it only once
	__global__ void kernel_calculate_face_normals_gather(Vertex* vertices, ReducedHalfEdge* half_edges, Loop* loops, float3* normals, unsigned loops_count) {
		int stride = thread_stride();
		int offset = thread_offset();

		for (int i = offset; i < loops_count; i += stride) {
			if (loops[i].is_border) {
				continue;
			}
			ReducedHalfEdge halfedge_a = half_edges[loops[i].he];
			ReducedHalfEdge halfedge_b = half_edges[halfedge_a.next];
			ReducedHalfEdge halfedge_c = half_edges[halfedge_b.next];
			float3 b = vertices[halfedge_b.origin].position;

			normals[i] = normalized(cross3df(b-vertices[halfedge_a.origin].position, vertices[halfedge_c.origin].position)-b);
		}
	}
	__global__ void kernel_calculate_normals_gather_from_loops(Vertex* vertices, ReducedHalfEdge* half_edges, Loop* loops, float3* face_normals, float3* normals, unsigned vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		for (int i = offset; i < vertice_count; i += stride) {
			const auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}
			float3 normal{ 0.f,0.f,0.f };
			int he = vert.he;
			do {//for every neighbor
				const ReducedHalfEdge& halfedge = half_edges[he];
				//skip boundary loops
				if (loops[halfedge.loop].is_border) {
					he = half_edges[halfedge.inv].next;
					continue;
				}
				normal += face_normals[halfedge.loop];
				he = half_edges[halfedge.inv].next;
			} while (he != vert.he);
			normals[i] = normalized(normal);
		}
	}

	//recalculates normals for every face
	__global__ void kernel_calculate_normals_gather(Vertex* vertices, HalfEdge* half_edges,Loop* loops, float3* normals, unsigned loops_count) {
		int stride = thread_stride();
		int offset = thread_offset();
		//calculate normals
		for (int i = offset; i < loops_count; i+=stride) {
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

	__global__ void kernel_calculate_ring_centroids_gather(Vertex* vertices, HalfEdge* half_edges, float3* centroids, unsigned vertice_count) {
		int stride = thread_stride();
		int offset = thread_offset();

		//calculate centroids
		for (int i = offset; i < vertice_count; i += stride) {
			auto& vert = vertices[i];
			//check for orphaned vertices
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
				centroid += vertices[inv_halfedge.origin].position;
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
			if (edge.first > -1) {
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

	__global__ void kernel_find_edges(pair<int,int>* pairs, int* faces, int* face_indices,int face_count) {
		int stride = thread_stride();
		int offset = thread_offset();

		for (int i = offset; i+1<face_count; i += stride) {
			int face_start = faces[i];
			int face_end = faces[i + 1];
			for (int j = face_start; j < face_end; ++j) {
				int first, second;
				//check for edge
				if (face_end == j + 1) {
					second = face_indices[face_start];
					first = face_indices[j];
				}
				else {
					first = face_indices[j];
					second = face_indices[j + 1];
				}

				if (first > second) {
					pairs[j] = pair<int, int>(second, first);
				}
				else {
					pairs[j] = pair<int, int>(first, second);
				}
			}
		}
	}

	struct PairLessThan {
		__device__  __host__ bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
			return a.first < b.first || (a.first == b.first && a.second < b.second);
		}
	};


	void normals_he_cuda_twopass_broken(HalfedgeMesh* mesh, int threads, int blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size()); //prepare vector for normals
		create_reduced_halfedges(*mesh);
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_gather);
		timing.block_size = threads;
		timing.grid_size = blocks;
		std::chrono::steady_clock::time_point start, stop, pstart, pstop;

		start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<ReducedHalfEdge> half_edges(mesh->reduced_half_edges.size());
		thrust::device_vector<Vertex> vertices(mesh->vertices.size());
		thrust::device_vector<Loop> loops(mesh->loops.size());
		thrust::device_vector<float3> normals(mesh->vertices.size());
		thrust::device_vector<float3> face_normals(loops.size(), { 0.f,0.f,0.f });
		cudaMemcpyAsync(half_edges.data().get(), mesh->reduced_half_edges.data(), mesh->reduced_half_edges.size() * sizeof(HalfEdge), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(vertices.data().get(), mesh->vertices.data(), mesh->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(loops.data().get(), mesh->loops.data(), mesh->loops.size() * sizeof(Loop), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//setup timer
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//kernel launch
		pstart = std::chrono::steady_clock::now();
		cudaEventRecord(cu_start);
		kernel_calculate_face_normals_gather<<<blocks, threads>>>(vertices.data().get(),
			half_edges.data().get(), loops.data().get(), face_normals.data().get(), loops.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		//cudaDeviceSynchronize();
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);

		cudaEventRecord(cu_start);
		kernel_calculate_normals_gather_from_loops<<<blocks, threads>>>(vertices.data().get(),
			half_edges.data().get(), loops.data().get(), face_normals.data().get(),normals.data().get(), vertices.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		//cudaDeviceSynchronize();
		pstop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);

		timing.processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(pstop - pstart).count();
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void normals_he_cuda(HalfedgeMesh* mesh, int threads,int blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->vertices.size()); //prepare vector for normals
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_gather);
		timing.block_size = threads;
		timing.grid_size = blocks;
		std::chrono::steady_clock::time_point start, stop, pstart, pstop;

		start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<HalfEdge> half_edges(mesh->half_edges.size());
		thrust::device_vector<Vertex> vertices(mesh->vertices.size());
		thrust::device_vector<Loop> loops(mesh->loops.size());
		thrust::device_vector<float3> normals(mesh->vertices.size());
		cudaMemcpyAsync(half_edges.data().get(), mesh->half_edges.data(), mesh->half_edges.size() * sizeof(HalfEdge), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(vertices.data().get(), mesh->vertices.data(), mesh->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(loops.data().get(), mesh->loops.data(), mesh->loops.size() * sizeof(Loop), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//setup timer
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//kernel launch
		pstart = std::chrono::steady_clock::now();
		cudaEventRecord(cu_start);
		kernel_calculate_normals_gather<<<blocks,threads>>>(vertices.data().get(), 
				half_edges.data().get(),loops.data().get(), normals.data().get(), vertices.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		//cudaDeviceSynchronize();
		pstop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		timing.processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(pstop - pstart).count();
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();//download time
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	/// normals from a simple mesh
	void normals_sm_cuda(SimpleMesh* mesh,int threads,int blocks, timing_struct& timing) {
		mesh->normals.resize(mesh->positions.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_normals_scatter);
		timing.block_size = threads;
		timing.grid_size = blocks;
		std::chrono::steady_clock::time_point start, stop, pstart, pstop;

		start = std::chrono::steady_clock::now(); //upload time
		thrust::device_vector<float3> positions(mesh->positions.size());
		thrust::device_vector<int> faces(mesh->faces.size());
		thrust::device_vector<int> face_indices(mesh->face_indices.size());
		thrust::device_vector<float3> normals(mesh->positions.size());
		cudaMemcpyAsync(positions.data().get(), mesh->positions.data(), mesh->positions.size() * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(faces.data().get(), mesh->faces.data(), mesh->faces.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(face_indices.data().get(), mesh->face_indices.data(), mesh->face_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//run kernel
		pstart = std::chrono::steady_clock::now();
		cudaEventRecord(cu_start);
		kernel_calculate_normals_scatter<<<blocks, threads>>>(positions.data().get(), faces.data().get(), face_indices.data().get(), normals.data().get(), faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//run secound kernel
		cudaEventRecord(cu_start);
		kernel_normalize_vectors<<<blocks, threads>>>(normals.data().get(),normals.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		pstop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);
		timing.processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(pstop - pstart).count();
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		start = std::chrono::steady_clock::now();
		//printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_he_cuda(HalfedgeMesh* mesh, attribute_vector<float3>& centroids_array, int threads,int blocks, timing_struct& timing) {
		centroids_array.resize(mesh->vertices.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_gather);
		timing.block_size = threads;
		timing.grid_size = blocks;

		std::chrono::steady_clock::time_point start, stop, pstart, pstop;

		start = std::chrono::steady_clock::now();
		thrust::device_vector<HalfEdge> half_edges(mesh->half_edges.size());
		thrust::device_vector<Vertex> vertices(mesh->vertices.size());
		thrust::device_vector<Loop> loops(mesh->loops.size());
		thrust::device_vector<float3> centroids(mesh->vertices.size(), { 0,0,0 });
		cudaMemcpyAsync(half_edges.data().get(), mesh->half_edges.data(), mesh->half_edges.size() * sizeof(HalfEdge), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(vertices.data().get(), mesh->vertices.data(), mesh->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(loops.data().get(), mesh->loops.data(), mesh->loops.size() * sizeof(Loop), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		//create events
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);
		//launch kernel
		pstart = std::chrono::steady_clock::now();
		cudaEventRecord(cu_start);
		kernel_calculate_ring_centroids_gather<<<blocks, threads>>>(vertices.data().get(), half_edges.data().get(), centroids.data().get(), vertices.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		pstop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		timing.processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(pstop - pstart).count();

		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);

		//read back
		start = std::chrono::steady_clock::now();
		thrust::copy(centroids.begin(), centroids.end(), centroids_array.begin());
		stop = std::chrono::steady_clock::now();
		timing.data_download_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	}

	void centroids_sm_cuda(SimpleMesh* mesh, attribute_vector<float3>& centroids_array, int threads,int blocks, timing_struct& timing) {
		centroids_array.resize(mesh->positions.size());
		if (threads == 0) optimal_configuration(blocks, threads, kernel_calculate_ring_centroids_scatter);
		timing.block_size = threads;
		timing.grid_size = blocks;
		std::chrono::steady_clock::time_point start, stop, pstart, pstop;
		//Data Upload Phase
		start = std::chrono::steady_clock::now();
		thrust::device_vector<float3> positions(mesh->positions.size());
		thrust::device_vector<int> faces(mesh->faces.size());
		thrust::device_vector<int> faces_indices(mesh->face_indices.size());
		thrust::device_vector<float3> centroids(mesh->positions.size(), { 0,0,0 });
		thrust::device_vector<int> neighbor_count(mesh->positions.size(), 0);
		thrust::device_vector<pair<int, int>> edges(faces_indices.size() - 1, pair<int, int>(-1, -1));//max size == edgecount <= face_indices - 1
		cudaMemcpyAsync(positions.data().get(), mesh->positions.data(), mesh->positions.size() * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(faces.data().get(), mesh->faces.data(), mesh->faces.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(faces_indices.data().get(), mesh->face_indices.data(), mesh->face_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		stop = std::chrono::steady_clock::now();
		timing.data_upload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();


		//Processing phase
		cudaEvent_t cu_start, cu_stop;
		cudaEventCreate(&cu_start);
		cudaEventCreate(&cu_stop);

		//prepare edge list
		cudaEventRecord(cu_start);
		kernel_find_edges<<<blocks,threads>>>(edges.data().get(), faces.data().get(), faces_indices.data().get(), faces.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_prepare = cuda_elapsed_time(cu_start, cu_stop);

		start = std::chrono::steady_clock::now();
		thrust::sort(thrust::device,reinterpret_cast<size_t*>(edges.data().get()), reinterpret_cast<size_t*>(edges.data().get()+edges.size()));
		stop = std::chrono::steady_clock::now();
		timing.sorting_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		start = std::chrono::steady_clock::now();
		auto last = thrust::unique(thrust::device,edges.begin(), edges.end());
		//edges.resize(last-edges.begin());
		stop = std::chrono::steady_clock::now();
		timing.unique_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

		//run kernel
		pstart = std::chrono::steady_clock::now();
		cudaEventRecord(cu_start);
		//kernel_calculate_ring_centroids_scatter_no_borders<<<blocks, threads>>>(positions.data().get(), faces.data().get(),
		//		faces_indices.data().get(), faces_sizes.data().get(), centroids.data().get(),neighbor_count.data().get(), faces.size());
		kernel_calculate_ring_centroids_scatter<<<blocks, threads>>>(positions.data().get(),edges.data().get(), centroids.data().get(),neighbor_count.data().get(), last - edges.begin());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		timing.kernel_execution_time_a = cuda_elapsed_time(cu_start, cu_stop);
		//divide
		cudaEventRecord(cu_start);
		kernel_divide<<<blocks, threads>>>(centroids.data().get(), neighbor_count.data().get(), centroids.size());
		cudaEventRecord(cu_stop);
		cudaEventSynchronize(cu_stop);
		pstop = std::chrono::steady_clock::now();
		timing.kernel_execution_time_b = cuda_elapsed_time(cu_start, cu_stop);
		timing.processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(pstop - pstart).count();
		cudaEventDestroy(cu_start);
		cudaEventDestroy(cu_stop);
		//copy back phase
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



