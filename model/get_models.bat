@echo off

cd %~dp0
powershell -Command "Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -OutFile sam_vit_h_4b8939.pth"
powershell -Command "Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -OutFile sam_vit_l_0b3195.pth"
powershell -Command "Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile sam_vit_b_01ec64.pth"

@pause