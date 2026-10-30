@echo off
rem Stop the lidar recording on the Pi, export its topics to CSV, and copy
rem both the bag and the CSVs to this laptop's data folder.
rem   Usage:  stop_and_fetch.cmd run3
setlocal
set RUN=%~1
if "%RUN%"=="" (
    echo Usage: %~nx0 run_name   e.g. %~nx0 run3
    exit /b 1
)
set DEST=%~dp0data

echo === Stopping recording on the Pi
ssh pi "~/lidar_motion/record.sh stop" || goto :fail

echo === Exporting topics to CSV on the Pi
ssh pi "source /opt/ros/jazzy/setup.bash && python3 ~/lidar_motion/export_topics.py ~/lidar_bags/%RUN%" || goto :fail

echo === Copying to %DEST%
if not exist "%DEST%" mkdir "%DEST%"
scp -r pi:~/lidar_bags/%RUN% pi:~/lidar_bags/%RUN%_csv "%DEST%" || goto :fail

echo === Done. Files on this laptop:
dir /b "%DEST%\%RUN%" "%DEST%\%RUN%_csv"
exit /b 0

:fail
echo *** Failed - see the message above. Nothing was deleted on the Pi.
exit /b 1
