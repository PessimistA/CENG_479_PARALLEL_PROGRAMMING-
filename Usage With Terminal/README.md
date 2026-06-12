# NLM Batch Benchmark CLI

## Overview
This tool is a command-line interface (CLI) for running Non-Local Means (NLM) image filtering tests on multiple images at the same time. It automatically finds all images in a specific folder, processes them using GPU acceleration (CUDA), and creates a summary of the results.

## Features
* **Batch Image Processing:** Automatically finds and processes `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.tga` files inside your selected folder.
* **Auto-Compilation:** The script compiles the CUDA codes automatically before starting the tests.
* **Data Merging:** It saves individual test results for each image and merges them into a single `all_results.json` file.
* **Summary Table:** Prints a clear, easy-to-read table directly in your terminal showing execution times, speedup factors, and error rates (MSE).

## System Requirements
* Linux Operating System
* NVIDIA GPU with CUDA Toolkit (`nvcc` compiler)
* Python 3.x

## How to Use Images and Run the Script

1. **Prepare Your Images:**
   Put all the medical images you want to test into a single folder. For example, create a folder named `my_images` and put all your image files inside it.

2. **Run the Script:**
   Open your terminal and make the script executable. Then, run the script by giving the image folder path as the first argument. You can also specify how many times to repeat the tests as the second argument (the default is 3).
   
   ```bash
   chmod +x run_benchmark.sh
   ./run_benchmark.sh /path/to/your/my_images 3
   ```

3. **View the Results:**
   When the script finishes, it will create a new folder named `benchmark_results` inside your images directory.
   * Inside this folder, there is a separate sub-folder for every image. You can see the filtered images there.
   * You will also find `all_results.json` containing all the merged mathematical data.
   * A summary table will be printed in your terminal, showing a fast comparison of all images.