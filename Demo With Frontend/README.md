# NLM Medical Image Filtering 

## Overview
This project provides a desktop application to analyze the performance of the Non-Local Means image denoising algorithm. It is designed specifically for medical images. The software evaluates the execution time differences between sequential CPU processing and parallel GPU processing using CUDA.

## Key Features
* **CUDA Acceleration:** Uses both Global Memory and Shared Memory optimizations for the GPU.
* **Live Benchmarking:** Shows real-time terminal output and live image updates while the C++ algorithms process the data.
* **Automated Block Size Sweep:** The system automatically tests 8x8, 16x16, and 32x32 thread blocks to find the fastest GPU configuration.
* **Data Visualization:** Automatically reads JSON output to generate detailed charts. These charts compare speedup, execution time, and workload limits.

## System Requirements
* Linux Operating System (Ubuntu, Pop!_OS, or similar distributions)
* NVIDIA GPU with CUDA Toolkit installed (`nvcc` compiler is required)
* G++ Compiler
* Python 3.x

## Installation and Build Instructions
The project includes an automated build script. This script compiles the C++ binaries, creates a Python virtual environment, installs the required libraries, and packages the interface into a standalone executable.

1. Open your terminal and make the script executable:
```bash
chmod +x build.sh
```

2. Run the build system:
```bash
./build.sh
```

3. Wait for the process to finish. When it is done, the final application and all necessary files will be placed in a new folder named `NLM_Release`.

## Usage
To start the application, navigate to the release folder and run the executable file:

```bash
cd NLM_Release
./app
```

**Using the Interface:**
1. Go to the "Run Benchmark & Live Viewer" tab.
2. Select your input image.
3. Choose your preferred GPU memory mode (Global or Shared).
4. Click "START LIVE BENCHMARK". 
5. Wait for the test to complete. You can watch the live terminal and image stream.
6. Once it finishes, switch to the "Evaluate Results & Graphs" tab.
7. Load the generated JSON file from your output folder to view the performance graphs and data tables.

**Extra Note - Viewing Past Results:** In the "Live Image Stream & Best Quality Viewer" section, you can click the **"Load Specific Folder"** button to choose a previously generated output folder. This allows you to load past benchmark data and manually select, view, and compare all the processed images from your earlier tests without having to run a new one.
