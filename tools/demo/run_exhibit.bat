@echo off
rem Exhibit mode launcher - double-click. Starts the WSL server and opens the browser full screen.
rem Stop: close this window or press Ctrl+C. Close the kiosk browser with Alt+F4.
rem Options are passed through, e.g. run_exhibit.bat --window
title ACVS exhibit
wsl.exe -d Ubuntu-22.04 --cd /home/yhjeon/Adaptive-Camera-View-Selection-System bash tools/demo/run_exhibit.sh %*
pause
