//#include <vtkm/cont/Initialize.h>
#include <boost/program_options.hpp>
#include <SimpleMesh.hpp>
#include <HalfEdgeMesh.hpp>
#include <iostream>

using namespace boost::program_options;
using namespace std;
using namespace ab;

int main(int argc, char* argv[]){
	//vtkm::cont:: Initialize(argc , argv);
	
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("in", value<string>(), "Ply File for reading")
			("out", value<string>(), "Ply File for writing");;

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
		else
			return 0;

		string fn = vm["in"].as<string>();
		SimpleMesh mesh;
		read_ply(mesh, fn);

		string sfn = fn + "-pos_only.ply";
		std::cout << "creating file: " << sfn << '\n';
		write_ply(mesh, sfn);


		HalfedgeMesh he_mesh;
		std::cout << "creating he mesh from simple mesh\n";
		create_he_mesh_from(he_mesh, mesh);

		SimpleMesh  copy_mesh;
		create_simple_mesh_from(copy_mesh, he_mesh);
		string hes_fn = fn + "-he-to-simple.ply";
		std::cout << "creating file: " << hes_fn << '\n';
		write_ply(copy_mesh, hes_fn);
	}
	catch (const error &ex)
	{
		std::cerr << ex.what() << '\n';
	}

	return 0;
}