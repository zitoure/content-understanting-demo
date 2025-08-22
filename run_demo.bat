@echo off
echo Azure AI Content Understanding Demo - Quick Start
echo ================================================

echo.
echo 1. Setting up environment...
python setup.py

echo.
echo 2. Running comprehensive demo...
python scenarios\comprehensive_demo.py

echo.
echo Demo completed! Check the output directory for results.
pause
