@ECHO OFF

SET OPROC=build\Release\onering.exe
REM SET OPROC=echo
SET CPU_ALGORITHMS=normals-gather-cpu normals-scatter-cpu centroids-gather-cpu centroids-scatter-cpu
SET CUDA_ALGORITHMS=normals-gather-cuda normals-scatter-cuda centroids-gather-cuda centroids-scatter-cuda
SET CUDA_BLOCK_CONFS=2 3 4 8

for %%f in ("%1\*.ply") do (
	for %%a in (%CPU_ALGORITHMS%) do (
		%OPROC% --alogorithm="%%a" --in="%%f" --out="%%~nf-%%a-(1,2).ply" --threads=2
		%OPROC% --alogorithm="%%a" --in="%%f" --out="%%~nf-%%a-(1,4).ply" --threads=4
		%OPROC% --alogorithm="%%a" --in="%%f" --out="%%~nf-%%a-(1,8).ply" --threads=8
		%OPROC% --alogorithm="%%a" --in="%%f" --out="%%~nf-%%a-(1,16).ply" --threads=16
	)
	for %%a in (%CUDA_ALGORITHMS%) do (
		for %%b in (%CUDA_BLOCK_CONFS%) do (
		%OPROC% --alogorithm="%%a" --in="%%f" --out="%%~nf-%%a-(%%b,256).ply" --threads=256 --blocks=%%b
		)
	)
)