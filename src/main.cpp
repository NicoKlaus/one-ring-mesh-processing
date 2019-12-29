//#include <vtkm/cont/Initialize.h>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>
#include <fast_mesh_operations.h>
#include <iostream>
#include <string>

using namespace boost::program_options;
using namespace std;
using namespace ab;

bool test_mesh(string fn) {
	SimpleMesh mesh;
	read_ply(mesh, fn);

	string sfn = fn + "-pos_only.ply";
	std::cout << "creating file: " << sfn << '\n';
	write_ply(mesh, sfn);


	HalfedgeMesh he_mesh;
	std::cout << "creating he mesh from simple mesh\n";
	create_he_mesh_from(he_mesh, mesh);

	{
		string hes_fn = fn + "-he-to-simple.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_ply(he_mesh, hes_fn)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		calculate_normals_he_seq(he_mesh);
		string hes_fn = fn + "-he-seq-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_ply(he_mesh, hes_fn)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	{
		std::cout << "calculate normals with cuda\n";
		calculate_normals_he_parallel_no_weight(&he_mesh);
		string hes_fn = fn + "-he-cuda-normals.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		if (!write_ply(he_mesh, hes_fn)) {
			std::cerr << "failed creating file: " << hes_fn << '\n';
		}
	}
	return true;
}

int main(int argc, char* argv[]){
	//vtkm::cont:: Initialize(argc , argv);
	
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("in", value<string>(), "Ply File for reading")
			("out", value<string>(), "Ply File for writing")
			("test", value<string>(), "Ply File for run tests on");

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