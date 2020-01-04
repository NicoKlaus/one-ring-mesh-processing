#include <fast_mesh_operations.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

using namespace thrust;

namespace ab {

	__global__ void kernel_calculate_normals_no_weight(Vertex* vertices, HalfEdge* half_edges, float3* normals, unsigned vertice_count) {
		int stride = blockDim.x;
		int offset = threadIdx.x;
		//printf("BLOCK %d launched by the host with stride %d\n", offset,stride);
		//calculate normal without weight
		for (int i = offset; i < vertice_count; i+=stride) {
			auto& vert = vertices[i];
			if (vert.he == -1) {
				continue;
			}
			int he = vert.he;
			float3 normal;
			normal.x = 0.f;
			normal.y = 0.f;
			normal.z = 0.f;
			do {
				HalfEdge& halfedge = half_edges[he];
				float3 a = vertices[halfedge.origin].position;
				float3 b = vertices[half_edges[halfedge.next].origin].position;
				normal = normal + cross3df(a, b);
				he = halfedge.next;
			} while (he != vert.he);
			normals[i] = normalized(normal);
		}
	}

	void calculate_normals_he_parallel_no_weight(HalfedgeMesh* mesh) {
		mesh->normals.resize(mesh->vertices.size());
		thrust::device_vector<HalfEdge> halfedges = mesh->half_edges;
		thrust::device_vector<Vertex> vertices = mesh->vertices;
		thrust::device_vector<float3> normals = mesh->normals;
		kernel_calculate_normals_no_weight<<<1,128>>>(vertices.data().get(), halfedges.data().get(), normals.data().get(), vertices.size());
		cudaDeviceSynchronize();
		printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
		thrust::copy(normals.begin(), normals.end(), mesh->normals.begin());
	}
}



