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
#include "mesh_normal_generator_vtkm.hpp"
#include <vtkm\cont\Initialize.h>
#include <vtkm\Types.h>
#include <vtkm\cont\ArrayHandle.h>
#include <vtkm\cont\DataSetBuilderExplicit.h>
using namespace boost::program_options;
using namespace ab;
using namespace std;
using namespace vtkm::cont;
using namespace vtkm;


int main(int argc, char* argv[]) {
	vtkm::cont::Initialize();
	int runs = 1;
	string in, out, time_log,algo_name;
	SimpleMesh smesh;
	shared_ptr<processing_functor> funct;
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("in", value<string>(), "Ply File for reading")
			("out", value<string>(), "Ply File for writing")
			("algorithm", value<string>(), "normals-vtkm|centroids-vtkm")
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
				if (algo_name == "normals-vtkm") {
					read_mesh(smesh, fn);
					funct = make_shared<mesh_normal_generator_vtkm>(&smesh);
				} else {
					cerr << "unknown algorithm! :" << algo_name << "\n";
				}
			}
		}
		else
			return 0;

	}
	catch (const error & ex)
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
			cout << "finished in " << funct->timings.total_execution_time() << " ns\n";
			timings.push_back(funct->timings);
		}

		stringstream log_data;
		for (timing_struct timing : timings) {
			log_data << "[" << algo_name << "]\n"
				<< "data_upload_time=" << timing.data_upload_time << "\n"
				<< "kernel_execution_time_a=" << timing.kernel_execution_time_a << "\n"
				<< "kernel_execution_time_b=" << timing.kernel_execution_time_b << "\n"
				<< "data_download_time=" << timing.data_download_time << "\n\n";
		}

		if (out.size()) {
			mesh_normal_generator_vtkm* normal_gen = dynamic_cast<mesh_normal_generator_vtkm*>(funct.get());
			if (normal_gen) {
				if (smesh.positions.size()) write_mesh(smesh, out);
			}
		}
		if (time_log.size()) {
			fstream fs;
			fs.open(time_log, ios_base::out);
			fs << log_data.str();
			fs.close();
			cout << "wrote results to " << time_log << '\n';
		}
		else {
			cout << "Results:\n" << log_data.str();
		}

		return 0;
	}
}
