#include <SimpleMesh.hpp>
#include <happly.h>
#include <fast_mesh_operations.h>

using namespace std;
using namespace happly;


namespace ab {

	bool read_ply(SimpleMesh &mesh, const string &file) {
		PLYData plyIn(file);
		plyIn.validate();
		std::vector<float> xPos = plyIn.getElement("vertex").getProperty<float>("x");
		std::vector<float> yPos = plyIn.getElement("vertex").getProperty<float>("y");
		std::vector<float> zPos = plyIn.getElement("vertex").getProperty<float>("z");

		std::vector<float3> &positions = mesh.positions;
		positions.resize(xPos.size());
		for (size_t i = 0; i < positions.size(); ++i) {
			positions[i].x = xPos[i];
			positions[i].y = yPos[i];
			positions[i].z = zPos[i];
		}
		mesh.faces = plyIn.getElement("face").getListProperty<int>("vertex_indices");
		return true;
	}

	bool write_ply(const SimpleMesh &mesh, const std::string &file) {
		PLYData plyOut;
		plyOut.addElement("vertex",mesh.positions.size());
		
		//positions
		vector<float> coordsX;
		for (auto pos : mesh.positions) {
			coordsX.push_back(pos.x);
		}
		plyOut.getElement("vertex").addProperty<float>("x",coordsX);
	
		for (size_t i = 0; i < mesh.positions.size(); ++i) {
			coordsX[i] = mesh.positions[i].y;
		}
		plyOut.getElement("vertex").addProperty<float>("y", coordsX);
		
		for (size_t i = 0; i < mesh.positions.size(); ++i) {
			coordsX[i] = mesh.positions[i].z;
		}
		plyOut.getElement("vertex").addProperty<float>("z", coordsX);
		
		if (mesh.normals.size() > 0 && mesh.normals.size() == mesh.positions.size()) {
			coordsX.resize(mesh.normals.size());
			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].x;
			}
			plyOut.getElement("vertex").addProperty<float>("nx", coordsX);

			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].y;
			}
			plyOut.getElement("vertex").addProperty<float>("ny", coordsX);

			for (size_t i = 0; i < mesh.normals.size(); ++i) {
				coordsX[i] = mesh.normals[i].z;
			}
			plyOut.getElement("vertex").addProperty<float>("nz", coordsX);
		}

		//faces
		plyOut.addElement("face", mesh.faces.size());
		plyOut.getElement("face").addListProperty("vertex_indices", mesh.faces);
		plyOut.write(file);
		return true;
	}

}