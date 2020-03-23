Thge program takes a mesh as .ply file and processes it with the method given over the --algorithm parameter.
If the --out parameter is given a .ply file with the result is created
Optinally the --time-log parameter can be used to create logs with performance metrics

Program Options
  -h [ --help ]          Help screen
  --in arg               Ply File for reading
  --out arg              Ply File for writing
  --test arg             runs function tests,= ply File used as input for
                         testing
  --algorithm arg        [normals-gather-<dev>|normals-scatter-<dev>|centroids-
                         gather-<dev>|centroids-scatter-<dev>]
                          replace <dev> with cuda or cpu
  --threads arg          threads per block, blocks and threads are determined
                         automatically if ommited
  --blocks arg           blocks in the grid, has no effect for cpu only
                         algorithms, determined automatically if --threads
                         ommited
  --runs arg             =N ,run calculation N times for extensive time
                         mesuring
  --time-log arg         saves timings to a log file
  --strip-attributes arg removes all attributes from the mesh except positions
                         and connectivity
  --sort arg             sort input mesh internaly by veretx degree and store
                         result in the file given by the option --out
Usage:
onering.exe  --algorithm="normals-gather-cuda" --in=bunny.ply --out=bunny-normals.ply --time-log=bunny-normals.log --threads=256 --blocks=36 --runs=10

Build Instructions

- requirements
	- boost libraries (already included)
	- cmake

For building boost open a command line and type
cd lib/<boost dir>
bootstrap
.\b2 --with-program_options --with-filesystem --with-system

for building the project change directory to build and run
cmake ..
