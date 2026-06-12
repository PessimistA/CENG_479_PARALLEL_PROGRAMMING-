#!/usr/bin/env bash

# Stop execution immediately if any command fails to prevent broken builds
set -e

echo "=========================================================="
echo "   NLM Medical Image Analyzer - MASTER BUILD SYSTEM"
echo "=========================================================="
echo ""

echo " [1/4] Compiling C++ and CUDA sources..."

# Compile the baseline sequential algorithm using standard g++ optimizations
echo "  -> Compiling Sequential (CPU)..."
g++ -O3 -o nlm_sequential nlm_sequential.cpp

# Compile the basic global memory CUDA implementation utilizing native architecture flags
echo "  -> Compiling Global Memory (GPU)..."
nvcc -O3 -arch=native -o nlm_global nlm_global.cu

# Compile the optimized shared memory CUDA implementation utilizing native architecture flags
echo "  -> Compiling Shared Memory (GPU)..."
nvcc -O3 -arch=native -o nlm_shared nlm_shared.cu

echo " C++/CUDA Compilation Successful!"
echo ""

echo " [2/4] Setting up Python Virtual Environment (venv)..."
# Create an isolated Python environment to manage dependencies locally
python3 -m venv nlm_env

# Activate the local virtual environment for subsequent pip installations
source nlm_env/bin/activate
echo " Virtual Environment active."
echo ""

echo " [3/4] Installing dependencies from requirements.txt..."
# Ensure the package installer itself is up to date before fetching project requirements
pip install --upgrade pip
pip install -r requirements.txt
echo " Dependencies installed."
echo ""

echo " [4/4] Building the Desktop App with PyInstaller..."
# Bundle the Python application into a single executable file without a background terminal window
pyinstaller --onefile --windowed --hidden-import PIL._tkinter_finder app.py
echo " Python App Built Successfully!"
echo ""

echo " Packaging everything into 'NLM_Release' folder..."
# Create a deployment directory to store all final executables
mkdir -p NLM_Release

# Move all compiled binary artifacts into the unified release folder
mv dist/app NLM_Release/
mv nlm_sequential NLM_Release/
mv nlm_global NLM_Release/
mv nlm_shared NLM_Release/

# Remove temporary build directories created during the PyInstaller bundling process
rm -rf build dist app.spec

# Exit the isolated Python environment
deactivate

echo "=========================================================="
echo "   INSTALLATION COMPLETELY FINISHED!"
echo "=========================================================="
echo "  Your portable application is ready in the 'NLM_Release' folder."
echo "  To start the application, run:"
echo ""
echo "  ./NLM_Release/app"
echo "=========================================================="