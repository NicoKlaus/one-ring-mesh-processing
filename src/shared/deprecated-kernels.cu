/*
#include <fast_mesh_operations.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

	__global__ void kernel_calculate_face_centroids_scatter(float3* positions,int* faces,int* face_indices,int* face_sizes, float3* centroids, int face_count) {
		int stride = blockDim.x;
		int offset = threadIdx.x;
		for (int i = offset; i < face_count; i += stride) {
			int base_index = faces[i];
			int face_size = face_sizes[i];
			
			float3 centroid;
			centroid.x = 0.f;
			centroid.y = 0.f;
			centroid.z = 0.f;
			//circulate trough the rest of the face and calculate the normal
			for (int j = 0;j< face_size;++j){
				float3 point = positions[face_indices[base_index+j]];
				//adding to the centroid vector
				centroid += point;
			}
			centroid.x /= face_size;
			centroid.y /= face_size;
			centroid.z /= face_size;
			centroids[i] = centroid;
		}
	}

	__global__ void kernel_calculate_face_centroids_gather(Vertex* vertices, HalfEdge* half_edges,Loop* loops, float3* centroids, unsigned loop_count) {
		int stride = blockDim.x;
		int offset = threadIdx.x;
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
*/