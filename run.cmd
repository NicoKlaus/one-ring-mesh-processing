@ECHO OFF

SET OPROC=build\Release\onering.exe
REM SET OPROC=echo
SET CPU_ALGORITHMS=normals-gather-cpu normals-scatter-cpu centroids-gather-cpu centroids-scatter-cpu
SET CUDA_ALGORITHMS=normals-gather-cuda normals-scatter-cuda centroids-gather-cuda centroids-scatter-cuda
SET CUDA_BLOCK_CONFS=8
SET DIR_OUT=%2
SET RUNS=10

for %%f in ("%1\*.ply") do (
	for %%a in (%CPU_ALGORITHMS%) do (
		%OPROC%  --algorithm="%%a" --in="%%f" --out="%DIR_OUT%\%%~nf-%%a-(1,8).ply" --time-log="%DIR_OUT%\%%~nf-%%a-(1,8).log" --threads=8 --runs=%RUNS%
	)
	for %%a in (%CUDA_ALGORITHMS%) do (
		REM use automatic launch configuration
		%OPROC%  --algorithm="%%a" --in="%%f" --out="%DIR_OUT%\%%~nf-%%a-(auto).ply" --time-log="%DIR_OUT%\%%~nf-%%a-(auto).log" --runs=%RUNS%
		REM test other configurations
		for %%b in (%CUDA_BLOCK_CONFS%) do (
		%OPROC%  --algorithm="%%a" --in="%%f" --out="%DIR_OUT%\%%~nf-%%a-(%%b,256).ply" --time-log="%DIR_OUT%\%%~nf-%%a-(%%b,256).log" --threads=256 --blocks=%%b --runs=%RUNS%
		)
	)
)