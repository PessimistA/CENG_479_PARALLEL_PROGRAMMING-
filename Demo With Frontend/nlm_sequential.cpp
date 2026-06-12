// nlm_sequential.cpp
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

// Resizes the original input image to the target resolution to maintain consistency during tests
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

// Define the workload configurations to test different algorithmic stress levels
struct NLMWorkload {
    int search_radius;
    int patch_radius;
};

static const NLMWorkload WORKLOADS[] = {
    {3, 1}, {5, 2}, {7, 2}, {9, 3}, {11, 3}
};
static const int N_WORKLOADS = 5;

// Smoothing parameter for the NLM algorithm
#define H_PARAM 40.0f
static const int RESOLUTIONS[]    = {256, 512, 1024, 2048};
static const int N_RES            = 4;
static const int DEFAULT_REPEATS  = 3;

// Baseline CPU implementation of the Non-Local Means algorithm
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
                    float w = std::exp(-patch_dist / h2);
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

// Extracts the filename from a given path to use in the JSON output
static std::string basename_no_ext(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

// Formats a vector of double values into a JSON-compatible array string
static std::string json_doubles(const std::vector<double>& v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) ss << ",";
        ss << v[i];
    }
    ss << "]";
    return ss.str();
}

// Appends the CPU-only benchmark results into a consolidated JSON file
static void append_json_record_cpu(const std::string& json_path, const std::string& image_name,
                                   int resolution, int search_radius, int patch_radius,
                                   double cpu_tp, const std::vector<double>& cpu_times, double cpu_avg)
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
    rec << "    \"cpu_throughput_MP_s\": " << cpu_tp << "\n";
    rec << "  }";

    size_t pos = content.rfind(']');
    bool has_entries = (content.find('{') != std::string::npos);
    std::string insert = (has_entries ? ",\n" : "\n") + rec.str() + "\n";
    content.insert(pos, insert);

    std::ofstream fout(json_path);
    fout << content;
}

// Executes a single benchmark scenario defining specific resolution and radii parameters
void run_benchmark_case(const unsigned char* src, int src_w, int src_h,
                        int target_res, int search_radius, int patch_radius,
                        const std::string& out_dir, const std::string& image_name,
                        const std::string& json_path, int repeats)
{
    const int W = target_res;
    const int H = target_res;
    const int N = W * H;
    const double megapixel = N / 1000000.0;

    std::cout << "\n  [" << W << "x" << H << "] CPU Workload (Search: " << search_radius
              << ", Patch: " << patch_radius << ") resizing...\n";

    std::vector<unsigned char> resized(N);
    resize_grayscale(src, src_w, src_h, resized.data(), W, H);

    // Save the resized input image only once per resolution to avoid disk clutter
    std::string noisy_path = out_dir + "/noisy_" + std::to_string(W) + ".png";
    struct stat buffer;
    if (stat(noisy_path.c_str(), &buffer) != 0) {
        stbi_write_png(noisy_path.c_str(), W, H, 1, resized.data(), W);
    }

    std::vector<unsigned char> cpu_out(N);
    std::vector<double> cpu_times;
    std::cout << "  [CPU] Running " << repeats << " time(s)...\n";

    // Execute the baseline CPU algorithm to establish the reference execution time
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        nlm_cpu(resized.data(), cpu_out.data(), W, H, search_radius, patch_radius, H_PARAM);
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
    stbi_write_png((out_dir + "/denoised_cpu" + suffix + ".png").c_str(), W, H, 1, cpu_out.data(), W);

    append_json_record_cpu(json_path, image_name, target_res, search_radius, patch_radius, cpu_tp, cpu_times, cpu_avg);
}

// Application entry point initializing the execution pipeline
int main(int argc, char** argv)
{
    // Validate command line arguments to ensure the input image and output directory are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_dir> [repeats]\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string out_dir    = argv[2];
    const int repeats = (argc >= 4) ? std::atoi(argv[3]) : DEFAULT_REPEATS;

    int src_w, src_h, channels;
    unsigned char* src = stbi_load(input_path.c_str(), &src_w, &src_h, &channels, 1);
    
    // Terminate execution if the image file cannot be read
    if (!src) {
        std::cerr << "Error: cannot open " << input_path << "\n";
        return 1;
    }

    std::string img_name  = basename_no_ext(input_path);
    std::string json_path = out_dir + "/results_cpu.json";

    std::cout << "==============================================\n";
    std::cout << "Sequential Baseline Execution\n";
    std::cout << "Image  : " << input_path << " (" << src_w << "x" << src_h << ")\n";
    std::cout << "Output : " << out_dir   << "\n";
    std::cout << "==============================================\n";

    // Loop through all predefined resolutions and workloads to perform the full benchmark suite
    for (int i = 0; i < N_RES; ++i) {
        for (int w = 0; w < N_WORKLOADS; ++w) {
            run_benchmark_case(src, src_w, src_h, RESOLUTIONS[i], WORKLOADS[w].search_radius, WORKLOADS[w].patch_radius, out_dir, img_name, json_path, repeats);
        }
    }

    stbi_image_free(src);
    return 0;
}