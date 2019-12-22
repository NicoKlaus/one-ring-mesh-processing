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
		write_ply(mesh, fn + "-pos_only.ply");

		HalfedgeMesh he_mesh;
		create_he_mesh_from(he_mesh, mesh);
	}
	catch (const error &ex)
	{
		std::cerr << ex.what() << '\n';
	}

	return 0;
}