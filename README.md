# NLM Medical Image Filtering - Project Repository

## Project Overview
This repository contains the implementation and performance analysis of the Non-Local Means image denoising algorithm for medical images. It compares standard sequential CPU processing against parallel GPU processing using CUDA. The project tests both Global Memory and Shared Memory optimizations.

The project is divided into two main environments: a command-line interface for testing and a graphical user interface for live demonstrations.

## Project Structure
Based on the repository layout, the files are organized as follows:

## Important Note
The real solution of this project is in the Demo With Frontend part. Both sequential and parallel solutions can be found on different files.
Terminal solution is given for the users that doesn't want to run python code.

```text
Root/
│
├── Demo With Frontend/
│   ├── app.py
│   ├── build.sh
│   ├── nlm_global.cu
│   ├── nlm_sequential.cpp
│   ├── nlm_shared.cu
│   ├── requirements.txt
│   ├── stb_image.h
│   ├── stb_image_write.h
│   └── README.md
│
├── Usage With Terminal/
│   ├── global_usage/
│   │   ├── images/
│   │   ├── nlm_benchmark.cu
│   │   ├── run_benchmark.sh
│   │   ├── stb_image.h
│   │   └── stb_image_write.h
│   ├── shared_usage/
│   │   ├── images/
│   │   ├── nlm_benchmark.cu
│   │   ├── run_benchmark.sh
│   │   ├── stb_image.h
│   │   └── stb_image_write.h
│   └── README.md
│
└── README.md
```

## 1. Demo With Frontend
This folder contains a Python tkinter graphical interface. It is designed for live demonstrations and individual image analysis. You can select an image, choose a specific GPU memory mode, and watch the processing happen in real-time. It automatically generates performance charts after the execution.

* **Primary Use Case:** Visualizing the filtering process, comparing images side-by-side, and generating automatic performance graphs.
* **Documentation:** [Click here to read the Frontend Documentation](./Demo%20With%20Frontend/README.md)

## 2. Usage With Terminal
This folder contains the core C++ and CUDA codes without any graphical interface. It is organized into `global_usage` and `shared_usage` sub-directories. It uses a Bash script to process multiple images in batches. It compiles the code, tests all images located in the `images` folder, merges the results, and prints a summary table.

* **Primary Use Case:** Running heavy batch tests on multiple images, recording raw performance data, and testing hardware limits.
* **Documentation:** [Click here to read the Terminal Documentation](./Usage%20With%20Terminal/README.md)

## General System Requirements
To run either version of this project, your system must meet these requirements:
* **Operating System:** Linux
* **Hardware:** NVIDIA GPU
* **Compiler:** CUDA Toolkit (`nvcc` compiler) and G++
* **Scripting:** Python 3.x
