@echo off

cd %~dp0
powershell -Command "Invoke-WebRequest https://fox-gieg.com/patches/github/n1ckfg/segment-anything/data/data.zip -OutFile data.zip"
powershell Expand-Archive data.zip -DestinationPath .
mv .\data\* .
rmdir data
del data.zip

@pause