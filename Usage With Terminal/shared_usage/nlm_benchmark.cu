#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Resizes the original input image to the target resolution to ensure consistency across different benchmark runs
static void resize_grayscale(const unsigned char* src, int src_w, int src_h,
                             unsigned char* dst, int dst_w, int dst_h)
{
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int sx = (int)(x * src_w / (float)dst_w);
            int sy = (int)(y * src_h / (float)dst_h);
            sx = std::min(sx, src_w - 1);
            sy = std::min(sy, src_h - 1);
            dst[y * dst_w + x] = src[sy * src_w + sx];
        }
    }
}

// Defines workload configurations to test different levels of computational stress
// The array is sorted from very light loads to heavy loads to test hardware boundaries
struct NLMWorkload {
    int search_radius;
    int patch_radius;
};

static const NLMWorkload WORKLOADS[] = {
    {3, 1},
    {5, 2},
    {7, 2},
    {9, 3},
    {11, 3}
};
static const int N_WORKLOADS = 5;

// Smoothing parameter applied within the Non-Local Means algorithm calculations
#define H_PARAM 40.0f

static const int RESOLUTIONS[]    = {256, 512, 1024, 2048};
static const int N_RES            = 4;
static const int DEFAULT_REPEATS  = 3;

// Predefined thread block dimensions used to sweep and identify the most efficient GPU execution layout
static const int BLOCK_SIZES[][2] = {{8,8},{16,16},{32,32}};
static const int N_BLOCK_SIZES    = 3;

// Structure to record execution times for specific block configurations
struct BlockResult {
    int bw;
    int bh;
    std::vector<double> times;
    double avg;
};

// Structure to evaluate the quality of the GPU-filtered image against the CPU-filtered baseline
struct Metrics {
    double mse;
    double max_abs_error;
};

// Baseline CPU implementation of the Non-Local Means algorithm
// This function runs sequentially and provides the reference execution time and quality ground truth
void nlm_cpu(const unsigned char* input, unsigned char* output,
             int width, int height,
             int search_radius, int patch_radius, float h)
{
    const float h2 = h * h;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float weight_sum    = 0.0f;
            float pixel_val_sum = 0.0f;
            
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    int sx = x + dx;
                    int sy = y + dy;
                    if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
                    
                    float patch_dist = 0.0f;
                    
                    for (int py = -patch_radius; py <= patch_radius; ++py) {
                        for (int px = -patch_radius; px <= patch_radius; ++px) {
                            int px1 = std::min(std::max(x  + px, 0), width  - 1);
                            int py1 = std::min(std::max(y  + py, 0), height - 1);
                            int px2 = std::min(std::max(sx + px, 0), width  - 1);
                            int py2 = std::min(std::max(sy + py, 0), height - 1);
                            float d = (float)input[py1 * width + px1]
                                    - (float)input[py2 * width + px2];
                            patch_dist += d * d;
                        }
                    }
                    float w = expf(-patch_dist / h2);
                    weight_sum    += w;
                    pixel_val_sum += w * (float)input[sy * width + sx];
                }
            }
            output[y * width + x] = (weight_sum > 0.0f)
                ? (unsigned char)(pixel_val_sum / weight_sum)
                : input[y * width + x];
        }
    }
}

// GPU Kernel utilizing Shared Memory optimization to reduce redundant global memory accesses
__global__ void nlm_kernel(const unsigned char* input, unsigned char* output,
                            int width, int height,
                            int search_radius, int patch_radius, float h)
{
    // Dynamically allocated shared memory array to store the image tile and its surrounding boundary halo
    extern __shared__ unsigned char s_image[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Calculate the required dimensions of the halo region for the search and patch operations
    int R = search_radius + patch_radius;
    int tile_w = blockDim.x + 2 * R;
    int tile_h = blockDim.y + 2 * R;

    // Determine the global top-left starting coordinate of the block including the halo area
    int b_x = blockIdx.x * blockDim.x - R;
    int b_y = blockIdx.y * blockDim.y - R;

    // Load the necessary data from global memory into the shared memory array
    // Threads collaborate to load an area larger than the block dimensions
    for (int i = ty; i < tile_h; i += blockDim.y) {
        for (int j = tx; j < tile_w; j += blockDim.x) {
            // Apply boundary clamping to prevent illegal memory accesses for edge pixels
            int global_x = min(max(b_x + j, 0), width - 1);
            int global_y = min(max(b_y + i, 0), height - 1);
            s_image[i * tile_w + j] = input[global_y * width + global_x];
        }
    }
    
    // Synchronize all threads in the block to ensure the shared memory tile is fully populated before processing
    __syncthreads();

    // Threads mapped outside the actual image boundaries can safely exit after participating in the memory load
    if (x >= width || y >= height) return;

    const float h2 = h * h;
    float weight_sum    = 0.0f;
    float pixel_val_sum = 0.0f;

    // Calculate the center coordinates for the current thread within the shared memory array
    int cx = tx + R;
    int cy = ty + R;

    // Perform the main algorithmic calculations entirely using the fast shared memory array
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            
            // Ensure the search window does not exceed the global image bounds
            int global_sx = x + dx;
            int global_sy = y + dy;
            if (global_sx < 0 || global_sx >= width || global_sy < 0 || global_sy >= height) continue;

            int sx = cx + dx;
            int sy = cy + dy;

            float patch_dist = 0.0f;
            for (int py = -patch_radius; py <= patch_radius; ++py) {
                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    // Min/max boundary checks are omitted here because the shared memory tile contains the clamped halo
                    float d = (float)s_image[(cy + py) * tile_w + (cx + px)]
                            - (float)s_image[(sy + py) * tile_w + (sx + px)];
                    patch_dist += d * d;
                }
            }
            float w = expf(-patch_dist / h2);
            weight_sum    += w;
            pixel_val_sum += w * (float)s_image[sy * tile_w + sx];
        }
    }
    
    // Write the final computed pixel value back to the global output array
    output[y * width + x] = (weight_sum > 0.0f)
        ? (unsigned char)(pixel_val_sum / weight_sum)
        : s_image[cy * tile_w + cx]; 
}

// Computes error metrics like Mean Squared Error to validate GPU output against the CPU reference
Metrics compute_metrics(const unsigned char* a, const unsigned char* b, int n)
{
    double sum_sq = 0.0;
    double max_ae = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = (double)a[i] - (double)b[i];
        sum_sq += diff * diff;
        double ae = std::fabs(diff);
        if (ae > max_ae) max_ae = ae;
    }
    Metrics m;
    m.mse = sum_sq / n;
    m.max_abs_error = max_ae;
    return m;
}

// Extracts the core filename from a file path string to use in output logs
static std::string basename_no_ext(const std::string& path)
{
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

// Converts a vector of double precision numbers into a formatted JSON array string
static std::string json_doubles(const std::vector<double>& v)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) ss << ",";
        ss << v[i];
    }
    ss << "]";
    return ss.str();
}

// Constructs a JSON object with the current benchmark results and appends it to the unified log file
static void append_json_record(const std::string& json_path,
                               const std::string& image_name,
                               int resolution,
                               int search_radius, int patch_radius,
                               double cpu_tp, double gpu_tp,
                               const std::vector<double>& cpu_times,
                               double cpu_avg,
                               const std::vector<BlockResult>& sweep,
                               double best_gpu_avg,
                               int best_bw, int best_bh,
                               double speedup, double amdahl_p,
                               const Metrics& m)
{
    std::ifstream fin(json_path);
    std::string content;
    if (fin.good()) {
        std::ostringstream ss;
        ss << fin.rdbuf();
        content = ss.str();
    }
    fin.close();
    if (content.empty()) content = "[]";

    std::ostringstream rec;
    rec << std::fixed << std::setprecision(6);
    rec << "  {\n";
    rec << "    \"image\": \""       << image_name << "\",\n";
    rec << "    \"resolution\": "    << resolution << ",\n";
    rec << "    \"search_radius\": " << search_radius << ",\n";
    rec << "    \"patch_radius\": "  << patch_radius << ",\n";
    rec << "    \"cpu_times_s\": "   << json_doubles(cpu_times) << ",\n";
    rec << "    \"cpu_avg_s\": "     << cpu_avg << ",\n";
    rec << "    \"cpu_throughput_MP_s\": " << cpu_tp << ",\n";
    rec << "    \"block_size_sweep\": [\n";
    for (size_t i = 0; i < sweep.size(); ++i) {
        const BlockResult& br = sweep[i];
        rec << "      { \"block\": \"" << br.bw << "x" << br.bh << "\","
            << " \"times_s\": " << json_doubles(br.times) << ","
            << " \"avg_s\": " << br.avg << " }";
        if (i + 1 < sweep.size()) rec << ",";
        rec << "\n";
    }
    rec << "    ],\n";
    rec << "    \"best_block_size\": \"" << best_bw << "x" << best_bh << "\",\n";
    rec << "    \"gpu_avg_s\": "         << best_gpu_avg << ",\n";
    rec << "    \"gpu_throughput_MP_s\": " << gpu_tp << ",\n";
    rec << "    \"speedup\": "           << speedup << ",\n";
    rec << "    \"amdahl_p\": "          << amdahl_p << ",\n";
    rec << "    \"mse\": "               << m.mse << ",\n";
    rec << "    \"max_abs_error\": "     << m.max_abs_error << "\n";
    rec << "  }";

    size_t pos = content.rfind(']');
    bool has_entries = (content.find('{') != std::string::npos);
    std::string insert = (has_entries ? ",\n" : "\n") + rec.str() + "\n";
    content.insert(pos, insert);

    std::ofstream fout(json_path);
    fout << content;
}

// Orchestrates a single benchmark case by resizing the image, running CPU tests, sweeping GPU block sizes, and logging data
void run_benchmark_case(const unsigned char* src, int src_w, int src_h,
                        int target_res, int search_radius, int patch_radius,
                        const std::string& out_dir,
                        const std::string& image_name,
                        const std::string& json_path,
                        int repeats)
{
    const int W = target_res;
    const int H = target_res;
    const int N = W * H;
    const double megapixel = N / 1000000.0;

    std::cout << "\n  [" << W << "x" << H << "] Workload (Search: " << search_radius 
              << ", Patch: " << patch_radius << ") resizing...\n";

    std::vector<unsigned char> resized(N);
    resize_grayscale(src, src_w, src_h, resized.data(), W, H);

    // Save the resized baseline image only once per resolution to avoid redundant file creation
    std::string noisy_path = out_dir + "/noisy_" + std::to_string(W) + ".png";
    struct stat buffer;
    if (stat(noisy_path.c_str(), &buffer) != 0) {
        stbi_write_png(noisy_path.c_str(), W, H, 1, resized.data(), W);
    }

    std::vector<unsigned char> cpu_out(N);
    std::vector<unsigned char> gpu_out(N);
    std::vector<unsigned char> best_gpu_out(N);

    std::vector<double> cpu_times;
    std::cout << "  [CPU] Running " << repeats << " time(s)...\n";

    // Measure the execution time of the sequential CPU algorithm
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        nlm_cpu(resized.data(), cpu_out.data(), W, H,
                search_radius, patch_radius, H_PARAM);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        cpu_times.push_back(elapsed);
        std::cout << "    run " << r+1 << ": " << elapsed << " s\n";
    }

    double cpu_avg = 0.0;
    for (double t : cpu_times) cpu_avg += t;
    cpu_avg /= (double)cpu_times.size();
    double cpu_tp = megapixel / cpu_avg;
    
    std::cout << "  [CPU] Average: " << cpu_avg << " s (Throughput: " << cpu_tp << " MP/s)\n";

    std::string suffix = "_" + std::to_string(W) + "_s" + std::to_string(search_radius) + "_p" + std::to_string(patch_radius);
    stbi_write_png((out_dir + "/denoised_cpu" + suffix + ".png").c_str(),
                   W, H, 1, cpu_out.data(), W);

    // Allocate device memory and transfer the resized input image to the GPU
    unsigned char* d_input  = nullptr;
    unsigned char* d_output = nullptr;
    cudaMalloc(&d_input,  N);
    cudaMalloc(&d_output, N);
    cudaMemcpy(d_input, resized.data(), N, cudaMemcpyHostToDevice);

    std::vector<BlockResult> sweep;
    double best_gpu_avg = 1e18;
    int    best_bw = 16;
    int    best_bh = 16;

    // Test multiple thread block configurations to determine the fastest execution setup
    for (int b = 0; b < N_BLOCK_SIZES; ++b) {
        int BW = BLOCK_SIZES[b][0];
        int BH = BLOCK_SIZES[b][1];

        std::cout << "  [GPU " << BW << "x" << BH << "] Running "
                  << repeats << " time(s)...\n";

        dim3 threads(BW, BH);
        dim3 blocks((W + BW - 1) / BW, (H + BH - 1) / BH);

        // Dynamically calculate the required shared memory size based on the block size and halo radius
        int R = search_radius + patch_radius;
        size_t smem_size = (BW + 2 * R) * (BH + 2 * R) * sizeof(unsigned char);

        BlockResult br;
        br.bw = BW;
        br.bh = BH;

        for (int r = 0; r < repeats; ++r) {
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);

            cudaEventRecord(ev_start);
            
            // Launch the kernel and pass the calculated shared memory size as the third configuration parameter
            nlm_kernel<<<blocks, threads, smem_size>>>(d_input, d_output, W, H,
                                            search_radius, patch_radius, H_PARAM);
            
            cudaEventRecord(ev_stop);
            
            // Synchronize the device to ensure precise kernel execution timing
            cudaEventSynchronize(ev_stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            br.times.push_back((double)(ms / 1000.0f));

            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);

            std::cout << "    run " << r+1 << ": " << (ms/1000.0f) << " s\n";
        }

        br.avg = 0.0;
        for (double t : br.times) br.avg += t;
        br.avg /= (double)br.times.size();
        std::cout << "  [GPU " << BW << "x" << BH << "] Average: "
                  << br.avg << " s\n";

        // Preserve the output image produced by the most efficient block size
        if (br.avg < best_gpu_avg) {
            best_gpu_avg = br.avg;
            best_bw      = BW;
            best_bh      = BH;
            cudaMemcpy(best_gpu_out.data(), d_output, N, cudaMemcpyDeviceToHost);
        }

        sweep.push_back(br);
    }

    // Release allocated GPU memory to prevent memory leaks during extended tests
    cudaFree(d_input);
    cudaFree(d_output);

    double gpu_tp = megapixel / best_gpu_avg;
    std::cout << "  Best block size: " << best_bw << "x" << best_bh
              << " (" << best_gpu_avg << " s, Throughput: " << gpu_tp << " MP/s)\n";

    stbi_write_png((out_dir + "/denoised_gpu" + suffix + ".png").c_str(),
                   W, H, 1, best_gpu_out.data(), W);

    Metrics met    = compute_metrics(cpu_out.data(), best_gpu_out.data(), N);
    double speedup = cpu_avg / best_gpu_avg;
    
    // Calculate the theoretical maximum parallelization limit according to Amdahl's Law
    double amdahl_p = 1.0 - (1.0 / speedup);

    std::cout << "  Speedup       : " << speedup          << "x\n";
    std::cout << "  Amdahl (p)    : " << amdahl_p * 100.0 << "%\n";
    std::cout << "  MSE           : " << met.mse            << "\n";
    std::cout << "  Max Abs Error : " << met.max_abs_error << "\n";

    append_json_record(json_path, image_name, target_res,
                       search_radius, patch_radius,
                       cpu_tp, gpu_tp,
                       cpu_times, cpu_avg,
                       sweep, best_gpu_avg,
                       best_bw, best_bh,
                       speedup, amdahl_p, met);
}

// Application entry point initializing the execution pipeline
int main(int argc, char** argv)
{
    // Ensure all required command line arguments are provided before proceeding
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_dir> [repeats]\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string out_dir    = argv[2];
    const int repeats = (argc >= 4) ? std::atoi(argv[3]) : DEFAULT_REPEATS;

    int src_w, src_h, channels;
    unsigned char* src = stbi_load(input_path.c_str(),
                                   &src_w, &src_h, &channels, 1);
    
    // Abort execution if the provided input image cannot be loaded
    if (!src) {
        std::cerr << "Error: cannot open " << input_path << "\n";
        return 1;
    }

    std::string img_name  = basename_no_ext(input_path);
    std::string json_path = out_dir + "/results.json";

    std::cout << "==============================================\n";
    std::cout << "Image  : " << input_path << " ("
              << src_w << "x" << src_h << ")\n";
    std::cout << "Output : " << out_dir   << "\n";
    std::cout << "Repeats: " << repeats   << "\n";
    std::cout << "Block sizes: 8x8, 16x16, 32x32\n";
    std::cout << "Workloads  : " << N_WORKLOADS << " different param sets\n";
    std::cout << "==============================================\n";

    // Iterate over predefined resolutions and workloads to conduct a comprehensive performance evaluation
    for (int i = 0; i < N_RES; ++i) {
        for (int w = 0; w < N_WORKLOADS; ++w) {
            run_benchmark_case(src, src_w, src_h,
                               RESOLUTIONS[i], 
                               WORKLOADS[w].search_radius, 
                               WORKLOADS[w].patch_radius,
                               out_dir, img_name,
                               json_path, repeats);
        }
    }

    stbi_image_free(src);
    std::cout << "\nAll results saved to " << json_path << "\n";
    return 0;
}