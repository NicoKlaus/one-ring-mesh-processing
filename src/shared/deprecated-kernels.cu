/*
	{
		std::cout << "calculate face centroids with cuda (gather)\n";
		vector<float3> centroids;
		auto time = ab::perf::execution_time([&] {calculate_face_centroids_he_parallel(&he_mesh, centroids); });
		std::cout << "calculated centroids in " << time.count() << "ns\n";
		string he_centroid_fn = fn + "-he-cuda-face-centroids.ply";
		std::cout << "creating file: " << he_centroid_fn << '\n';
		write_pointcloud(he_centroid_fn, centroids.data(), centroids.size());
	}
	{
		std::cout << "calculate face centroids with cuda (scatter)\n";
		vector<float3> centroids;
		auto time = ab::perf::execution_time([&] {calculate_face_centroids_sm_parallel(&mesh, centroids); });
		std::cout << "calculated centroids in " << time.count() << "ns\n";
		string sm_centroid_fn = fn + "-sm-cuda-face-centroids.ply";
		std::cout << "creating file: " << sm_centroid_fn << '\n';
		write_pointcloud(sm_centroid_fn, centroids.data(), centroids.size());
	}
*/


/*header

	void calculate_face_centroids_sm_parallel(SimpleMesh* mesh, std::vector<float3>& centroids_array, size_t threads = 256, size_t blocks = 1);
	void calculate_face_centroids_he_parallel(HalfedgeMesh* mesh, std::vector<float3>& centroids_array, size_t threads = 256, size_t blocks = 1);
*/

/*source
#include <fast_mesh_operations.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>


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
*/