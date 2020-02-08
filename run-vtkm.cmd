@ECHO OFF

SET OPROC=build\Release\onering_vtkm.exe
REM SET OPROC=echo
SET CPU_ALGORITHMS=normals-vtkm
SET DIR_OUT=%2
SET RUNS=20

for %%f in ("%1\*.ply") do (
	for %%a in (%CPU_ALGORITHMS%) do (
		%OPROC%  --algorithm="%%a" --in="%%f" --out="%DIR_OUT%\%%~nf-%%a-(vtkm).ply" --time-log="%DIR_OUT%\%%~nf-%%a-(vtkm).log" --runs=%RUNS%
	)
)