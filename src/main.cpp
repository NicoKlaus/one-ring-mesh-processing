//#include <vtkm/cont/Initialize.h>
#include <perf.hpp>
#include <util.hpp>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>
#include <cuda_mesh_operations.hpp>
#include <cpu_mesh_operations.hpp>
#include <iostream>
#include <string>

using namespace boost::program_options;
using namespace std;
using namespace ab;

const bool bin_mode = 
#ifdef NDEBUG
true;
#else
false;
#endif

bool test_mesh(string fn,bool mesh_conversion_output = false) {
	SimpleMesh mesh;
	read_ply(mesh, fn);

	string sfn = fn + "-pos_only.ply";
	std::cout << "creating file: " << sfn << '\n';
	write_mesh(mesh, sfn,bin_mode);


	HalfedgeMesh he_mesh;
	std::cout << "creating he mesh from simple mesh\n";
	create_he_mesh_from(he_mesh, mesh);


	if (mesh_conversion_output){
		string hes_fn = fn + "-he-to-simple.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_mesh(he_mesh, hes_fn, bin_mode)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate normals with cpu (halfedge mesh)\n";
		auto time = ab::perf::execution_time([&]{normals_by_area_weight_he_cpu(&he_mesh,8);});
		std::cout << "calculated normals in " << time.count() << "ns\n";
		string hes_fn = fn + "-he-cpu-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_mesh(he_mesh, hes_fn, bin_mode)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate normals with cpu (simple mesh)\n";
		auto time = ab::perf::execution_time([&] {normals_by_area_weight_sm_cpu(&mesh, 8); });
		std::cout << "calculated normals in " << time.count() << "ns\n";
		string hes_fn = fn + "-sm-cpu-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_mesh(he_mesh, hes_fn, bin_mode)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate normals with cuda (gather)\n";
		he_mesh.normals.clear();
		auto time = ab::perf::execution_time([&]{calculate_normals_he_parallel_area_weight(&he_mesh); });
		std::cout << "calculated normals in " << time.count() << "ns\n";
		string hes_fn = fn + "-he-cuda-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_mesh(he_mesh, hes_fn, bin_mode)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate normals with cuda (scatter)\n";
		mesh.normals.clear();
		auto time = ab::perf::execution_time([&]{calculate_normals_sm_parallel_area_weight(&mesh); });
		std::cout << "calculated normals in " << time.count() << "ns\n";
		string hes_fn = fn + "-sm-cuda-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_mesh(mesh, hes_fn, bin_mode)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate one ring centroids with cuda (gather)\n";
		vector<float3> centroids;
		auto time = ab::perf::execution_time([&]{calculate_centroids_he_parallel(&he_mesh,centroids); });
		std::cout << "calculated centroids in " << time.count() << "ns\n";
		string he_centroid_fn = fn + "-he-cuda-centroids.ply";
		std::cout << "creating file: " << he_centroid_fn << '\n';
		write_pointcloud(he_centroid_fn, centroids.data(), centroids.size());
	}
	{
		std::cout << "calculate one ring centroids with cuda (scatter)\n";
		vector<float3> centroids;
		auto time = ab::perf::execution_time([&] {calculate_centroids_sm_parallel(&mesh, centroids); });
		std::cout << "calculated centroids in " << time.count() << "ns\n";
		string sm_centroid_fn = fn + "-sm-cuda-centroids.ply";
		std::cout << "creating file: " << sm_centroid_fn << '\n';
		write_pointcloud(sm_centroid_fn, centroids.data(), centroids.size());
	}
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
	return true;
}

int main(int argc, char* argv[]){
	
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("in", value<string>(), "Ply File for reading")
			("out", value<string>(), "Ply File for writing")
			("test", value<string>(), "runs tests,= Ply File for testing")
			("algorithm", value<string>(), "normals-gather-cuda|normals-scatter-cuda|centroids-gather-cuda|centroids-scatter-cuda")
			("runs", value<string>(), "=N ,run calculation N times for time mesuring");

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);


		string filename;
		if (vm.count("help")) {
			std::cout << desc << '\n';
			return 0;
		}
		else if (vm.count("in")) {
			std::cout << "reading file: " << vm["in"].as<string>() << '\n';
			if (vm.count("algorithm")) {
				string algo_name = vm["algorithm"].as<string>();
				size_t off;
				if (algo_name == "normals-gather-cpu") {

				}
				else if (algo_name == "normals-scatter-cpu") {

				}
				else if (algo_name == "normals-gather-cuda") {

				}
				else if (algo_name == "normals-scatter-cuda") {

				}
				else if (algo_name == "centroids-gather-cpu") {

				}
				else if (algo_name == "centroids-scatter-cpu") {

				}
				else if (algo_name == "centroids-gather-cuda") {

				}
				else if (algo_name == "centroids-scatter-cuda") {

				}
				else {
					cerr << "unknown algorithm! :" << algo_name << "\n";
				}
			}
		}
		else if (vm.count("test")) {
			test_mesh(vm["test"].as<string>());
		}
		else
			return 0;

	}
	catch (const error &ex)
	{
		std::cerr << ex.what() << '\n';
	}

	return 0;
}