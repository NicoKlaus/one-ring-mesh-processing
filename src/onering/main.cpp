//#include <vtkm/cont/Initialize.h>
#include <perf.hpp>
#include <util.hpp>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>
#include <mesh_centroid_generator.hpp>
#include <mesh_normal_generator.hpp>
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
	read_mesh(mesh, fn);

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
		auto time = ab::perf::execution_time([&]{normals_by_area_weight_he_cuda(&he_mesh); });
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
		auto time = ab::perf::execution_time([&]{normals_by_area_weight_sm_cuda(&mesh); });
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
		auto time = ab::perf::execution_time([&]{centroids_he_cuda(&he_mesh,centroids); });
		std::cout << "calculated centroids in " << time.count() << "ns\n";
		string he_centroid_fn = fn + "-he-cuda-centroids.ply";
		std::cout << "creating file: " << he_centroid_fn << '\n';
		write_pointcloud(he_centroid_fn, centroids.data(), centroids.size());
	}
	{
		std::cout << "calculate one ring centroids with cuda (scatter)\n";
		vector<float3> centroids;
		auto time = ab::perf::execution_time([&] {centroids_sm_cuda(&mesh, centroids); });
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
	
	std::shared_ptr<processing_functor> funct;
	SimpleMesh smesh;
	HalfedgeMesh hemesh;
	int threads = 8, blocks = 1;
	size_t runs = 1;
	string algo_name;
	string out;
	string time_log;
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("in", value<string>(), "Ply File for reading")
			("out", value<string>(), "Ply File for writing")
			("test", value<string>(), "runs tests,= Ply File for testing")
			("algorithm", value<string>(), "normals-gather-cuda|normals-scatter-cuda|centroids-gather-cuda|centroids-scatter-cuda")
			("threads", value<int>(), "threads per block")
			("blocks", value<int>(), "cuda blocks to start, has no effect for cpu only algorithms")
			("runs", value<int>(), "=N ,run calculation N times for time mesuring")
			("time-log", value<string>(), "saves timings to file");

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
			string fn = vm["in"].as<string>();

			if (vm.count("threads")) {
				threads = vm["threads"].as<int>();
			}
			if (vm.count("blocks")) {
				blocks = vm["blocks"].as<int>();
			}
			if (vm.count("runs")) {
				runs = vm["runs"].as<int>();
			}
			if (vm.count("out")) {
				out = vm["out"].as<string>();
			}
			if (vm.count("time-log")) {
				time_log = vm["time-log"].as<string>();
			}

			if (vm.count("algorithm")) {
				algo_name = vm["algorithm"].as<string>();
				size_t off;
				if (algo_name == "normals-gather-cpu") {
					read_mesh(hemesh, fn);
					funct = make_shared<mesh_normal_generator>(&hemesh, PD_CPU, threads, blocks);
				}
				else if (algo_name == "normals-scatter-cpu") {
					read_mesh(smesh, fn);
					funct = make_shared<mesh_normal_generator>(&smesh, PD_CPU, threads, blocks);
				}
				else if (algo_name == "normals-gather-cuda") {
					read_mesh(hemesh, fn);
					funct = make_shared<mesh_normal_generator>(&hemesh, PD_CUDA, threads, blocks);
				}
				else if (algo_name == "normals-scatter-cuda") {
					read_mesh(smesh, fn);
					funct = make_shared<mesh_normal_generator>(&smesh, PD_CUDA, threads, blocks);
				}
				else if (algo_name == "centroids-gather-cpu") {
					read_mesh(hemesh, fn);
					funct = make_shared<mesh_centroid_generator>(&hemesh, PD_CPU, threads, blocks);
				}
				else if (algo_name == "centroids-scatter-cpu") {
					read_mesh(smesh, fn);
					funct = make_shared<mesh_centroid_generator>(&smesh, PD_CPU, threads, blocks);
				}
				else if (algo_name == "centroids-gather-cuda") {
					read_mesh(hemesh, fn);
					funct = make_shared<mesh_centroid_generator>(&hemesh, PD_CUDA, threads, blocks);
				}
				else if (algo_name == "centroids-scatter-cuda") {
					read_mesh(smesh, fn);
					funct = make_shared<mesh_centroid_generator>(&smesh, PD_CUDA, threads, blocks);
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

	if (funct) {
		//run selected algorithm
		cout << "selected algorithm: " << algo_name << '\n';
		std::vector<timing_struct> timings;
		for (int i = 0; i < runs; ++i) {
			cout << "start run " << i << "/" << runs << " ...";
			(*funct)();
			cout << "finished in "<< funct->timings.total_execution_time() << " ns\n";
			timings.push_back(funct->timings);
		}

		stringstream log_data;
		for (timing_struct timing : timings) {
			log_data << "["<< algo_name << " threads=" << threads << " blocks=" << blocks << "]\n"
				<< "data_upload_time=" << timing.data_upload_time << "\n"
				<< "kernel_execution_time_a=" << timing.kernel_execution_time_a << "\n"
				<< "kernel_execution_time_b=" << timing.kernel_execution_time_b << "\n"
				<< "data_download_time=" << timing.data_download_time << "\n\n";
		}
		
		if (out.size()) {
			mesh_normal_generator* normal_gen = dynamic_cast<mesh_normal_generator*>(funct.get());
			if (normal_gen) {
				if (hemesh.half_edges.size()) write_mesh(hemesh, out);
				if (smesh.positions.size()) write_mesh(smesh, out);
			}
			mesh_centroid_generator* centroid_gen = dynamic_cast<mesh_centroid_generator*>(funct.get());
			if (centroid_gen) {
				write_pointcloud(out, centroid_gen->centroids.data(), centroid_gen->centroids.size());
			}
		} 
		if (time_log.size()) {
			fstream fs;
			fs.open(time_log, ios_base::out);
			fs << log_data.str();
			fs.close();
			cout << "wrote results to " << time_log << '\n';
		} else {
			cout << "Results:\n" << log_data.str();
		}

		return 0;
	}
}