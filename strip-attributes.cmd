@ECHO OFF

SET OPROC=build\Release\onering.exe
REM SET OPROC=echo
SET DIR_OUT=%2

for %%f in ("%1\*.ply") do (
	%OPROC% --strip-attributes=on --in="%%f" --out="%DIR_OUT%\%%~nf-pos-only.ply"
)