#include <util.hpp>

using namespace std;
using namespace happly;
namespace ab {
	void write_pointcloud(const string& fn, float3* points, size_t size)
	{
		PLYData plyOut;
		plyOut.addElement("vertex", size);

		//positions
		vector<float> coordsX;
		for (size_t i = 0; i < size; ++i) {
			coordsX.push_back(points[i].x);
		}
		plyOut.getElement("vertex").addProperty<float>("x", coordsX);

		for (size_t i = 0; i < size; ++i) {
			coordsX[i] = points[i].y;
		}
		plyOut.getElement("vertex").addProperty<float>("y", coordsX);

		for (size_t i = 0; i < size; ++i) {
			coordsX[i] = points[i].z;
		}
		plyOut.getElement("vertex").addProperty<float>("z", coordsX);
		plyOut.write(fn);
	}
}