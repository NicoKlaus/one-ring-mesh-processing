@ECHO OFF

SET OPROC=build\Release\onering.exe
REM SET OPROC=echo
SET CPU_ALGORITHMS=normals-gather-cpu normals-scatter-cpu centroids-gather-cpu centroids-scatter-cpu
SET CUDA_ALGORITHMS=normals-gather-cuda normals-scatter-cuda centroids-gather-cuda centroids-scatter-cuda
SET CUDA_BLOCK_CONFS=2 3 4 8
SET DIR_OUT=%2
SET RUNS=20

for %%f in ("%1\*.ply") do (
	for %%a in (%CPU_ALGORITHMS%) do (
		%OPROC%  --algorithm="%%a" --in="%%f" --time-log="%DIR_OUT%\%%~nf-%%a-(1,4).log" --threads=4 --runs=%RUNS%
		%OPROC%  --algorithm="%%a" --in="%%f" --time-log="%DIR_OUT%\%%~nf-%%a-(1,8).log" --threads=8 --runs=%RUNS%
	)
	for %%a in (%CUDA_ALGORITHMS%) do (
		for %%b in (%CUDA_BLOCK_CONFS%) do (
		%OPROC%  --algorithm="%%a" --in="%%f"  --time-log="%DIR_OUT%\%%~nf-%%a-(%%b,256).log" --threads=256 --blocks=%%b --runs=%RUNS%
		)
	)
)