import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import json
import os
import threading
import re
import glob
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NLMBenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLM Medical Image Filtering - Performance Analyzer")
        self.root.geometry("1400x900")
        
        # Bind the window close event to prevent background C++ processes from running indefinitely
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure the visual theme and default font styles for the UI components
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        
        # Create the main notebook widget to manage multiple tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize the frames that will act as individual tabs
        self.tab_run = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_run, text="Run Benchmark & Live Viewer")
        self.notebook.add(self.tab_eval, text="Evaluate Results & Graphs")
        
        self.current_out_dir = ""
        
        # This list prevents the Python garbage collector from deleting Tkinter image objects
        self.image_references = [] 
        
        # Variables to track and control the running C++ benchmark process
        self.current_process = None
        self.stop_requested = False
        
        self.setup_run_tab()
        self.setup_eval_tab()

    def on_closing(self):
        # Ensure all subprocesses are terminated before destroying the main window
        self.stop_benchmark(silent=True)
        self.root.destroy()

    def _bind_mouse_scroll(self, widget, canvas):
        # Enable mouse wheel scrolling only when the cursor is over the specific widget
        widget.bind("<Enter>", lambda e: self.root.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units")))
        widget.bind("<Leave>", lambda e: self.root.unbind_all("<MouseWheel>"))

    def setup_run_tab(self):
        # Create a split view: configuration on the left, live images on the right
        run_paned = ttk.PanedWindow(self.tab_run, orient=tk.HORIZONTAL)
        run_paned.pack(fill='both', expand=True)

        left_frame = ttk.Frame(run_paned)
        right_frame = ttk.Frame(run_paned)
        run_paned.add(left_frame, weight=1)
        run_paned.add(right_frame, weight=2)

        # Build the configuration panel to gather user inputs
        control_frame = ttk.LabelFrame(left_frame, text="Test Configuration")
        control_frame.pack(fill='x', padx=5, pady=5)

        # Input image selection path
        ttk.Label(control_frame, text="Input Image:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.img_path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.img_path_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_image).grid(row=0, column=2, padx=5, pady=5)

        # Output directory selection path
        ttk.Label(control_frame, text="Output Dir:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.out_dir_var = tk.StringVar(value="./benchmark_results")
        ttk.Entry(control_frame, textvariable=self.out_dir_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_out_dir).grid(row=1, column=2, padx=5, pady=5)

        # Memory mode selection dropdown
        ttk.Label(control_frame, text="Mode:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.mode_var = tk.StringVar(value="Parallel (GPU Shared Memory)")
        
        modes = [
            "Parallel (GPU Global Memory)", 
            "Parallel (GPU Shared Memory)"
        ]
        ttk.Combobox(control_frame, textvariable=self.mode_var, values=modes, state="readonly", width=27).grid(row=2, column=1, padx=5, pady=5)

        # Group execution control buttons together
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=3, column=1, pady=10, sticky='w')

        self.btn_run = ttk.Button(btn_frame, text="START LIVE BENCHMARK", command=self.start_benchmark_thread)
        self.btn_run.pack(side='left', padx=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="STOP", command=self.stop_benchmark, state="disabled")
        self.btn_stop.pack(side='left', padx=5)
        
        self.lbl_status = ttk.Label(control_frame, text="Status: Ready", foreground="blue", font=("Segoe UI", 10, "italic"))
        self.lbl_status.grid(row=4, column=1, pady=5, sticky='w')

        # Build the terminal output box to display C++ standard output
        console_frame = ttk.LabelFrame(left_frame, text="Live Terminal Output (C++ / CUDA)")
        console_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.txt_console = tk.Text(console_frame, bg="black", fg="lime green", font=("Consolas", 9), wrap=tk.WORD)
        console_scroll = ttk.Scrollbar(console_frame, command=self.txt_console.yview)
        self.txt_console.configure(yscrollcommand=console_scroll.set)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Build the image viewer area to show original, CPU, and GPU results
        viewer_frame = ttk.LabelFrame(right_frame, text="Live Image Stream & Best Quality Viewer")
        viewer_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.lbl_current_view = ttk.Label(viewer_frame, text="Live View: Waiting for test...", font=("Segoe UI", 12, "bold"), foreground="navy")
        self.lbl_current_view.pack(pady=5)

        # Controls to select previous benchmark results
        top_viewer_controls = ttk.Frame(viewer_frame)
        top_viewer_controls.pack(fill='x', pady=5)
        
        ttk.Label(top_viewer_controls, text="Select Result:").pack(side='left', padx=5)
        self.config_var = tk.StringVar()
        self.combo_config = ttk.Combobox(top_viewer_controls, textvariable=self.config_var, state="readonly", width=55)
        self.combo_config.pack(side='left', padx=5)
        
        # Automatically update displayed images when a new option is selected
        self.combo_config.bind("<<ComboboxSelected>>", self.update_image_display)
        ttk.Button(top_viewer_controls, text="Load Specific Folder", command=self.load_folder_directly).pack(side='left', padx=5)

        # Set up a scrollable canvas to hold large output images
        self.img_canvas = tk.Canvas(viewer_frame)
        self.img_scrollbar = ttk.Scrollbar(viewer_frame, orient="vertical", command=self.img_canvas.yview)
        
        self.img_canvas_frame = ttk.Frame(self.img_canvas)
        self.img_canvas_frame.bind("<Configure>", lambda e: self.img_canvas.configure(scrollregion=self.img_canvas.bbox("all")))
        self.img_canvas_window = self.img_canvas.create_window((0, 0), window=self.img_canvas_frame, anchor="nw")
        
        # Keep the canvas frame width synchronized with the canvas width
        self.img_canvas.bind("<Configure>", lambda e: self.img_canvas.itemconfig(self.img_canvas_window, width=e.width))
        self.img_canvas.configure(yscrollcommand=self.img_scrollbar.set)
        
        self.img_scrollbar.pack(side="right", fill="y")
        self.img_canvas.pack(side="left", fill="both", expand=True)
        
        # Attach custom scroll binding to the canvas area
        self._bind_mouse_scroll(self.img_canvas, self.img_canvas)

        # Placeholders for the result images
        self.lbl_orig_title = ttk.Label(self.img_canvas_frame, text="Original Resized Image", style="Header.TLabel")
        self.lbl_orig_title.grid(row=0, column=0, columnspan=2, pady=5)
        self.lbl_orig_img = ttk.Label(self.img_canvas_frame)
        self.lbl_orig_img.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.lbl_cpu_title = ttk.Label(self.img_canvas_frame, text="Filtered (CPU)", style="Header.TLabel")
        self.lbl_cpu_title.grid(row=2, column=0, pady=5)
        self.lbl_gpu_title = ttk.Label(self.img_canvas_frame, text="Filtered (GPU - Fastest Block)", style="Header.TLabel")
        self.lbl_gpu_title.grid(row=2, column=1, pady=5)
        
        self.lbl_cpu_img = ttk.Label(self.img_canvas_frame)
        self.lbl_cpu_img.grid(row=3, column=0, padx=10, pady=5)
        
        self.lbl_gpu_img = ttk.Label(self.img_canvas_frame)
        self.lbl_gpu_img.grid(row=3, column=1, padx=10, pady=5)
        
        self.img_canvas_frame.columnconfigure(0, weight=1)
        self.img_canvas_frame.columnconfigure(1, weight=1)

    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if filename: self.img_path_var.set(filename)

    def browse_out_dir(self):
        dirname = filedialog.askdirectory()
        if dirname: self.out_dir_var.set(dirname)

    def append_console(self, text):
        # Insert text at the end of the text widget and scroll down automatically
        self.txt_console.insert(tk.END, text)
        self.txt_console.see(tk.END)

    def stop_benchmark(self, silent=False):
        # Mark the stop request flag so running loops can break early
        self.stop_requested = True
        
        # Send a termination signal directly to the OS process if it is still active
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate() 
            if not silent:
                self.append_console("\n[!] PROCESS TERMINATED BY USER.\n")
                self.lbl_status.config(text="Status: Stopped by user.", foreground="red")
        
        if not silent:
            self.reset_run_btn()

    def start_benchmark_thread(self):
        # Validate the input file before attempting execution
        if not self.img_path_var.get() or not os.path.exists(self.img_path_var.get()):
            messagebox.showerror("Error", "Please select a valid input image.")
            return
            
        self.stop_requested = False
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.txt_console.delete(1.0, tk.END) 
        self.lbl_status.config(text="Status: Running benchmark... Watch terminal below.", foreground="red")
        
        # Clear old images while waiting for new results
        self.lbl_orig_img.config(image='', text="Waiting...")
        self.lbl_cpu_img.config(image='', text="Waiting...")
        self.lbl_gpu_img.config(image='', text="Waiting...")
        self.image_references.clear()

        # Run the executable in a background thread to prevent UI freezing
        threading.Thread(target=self.run_benchmark, daemon=True).start()

    def run_benchmark(self):
        img_path = self.img_path_var.get()
        base_out_dir = self.out_dir_var.get()
        mode = self.mode_var.get()
        
        # Create a specific sub-folder using the image's base name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(base_out_dir, img_name)
        os.makedirs(out_dir, exist_ok=True)
        self.current_out_dir = out_dir

        # Determine which binary to call based on the user's dropdown selection
        exe = "./nlm_shared" if "Shared" in mode else "./nlm_global"

        if not os.path.exists(exe):
            self.root.after(0, lambda: messagebox.showerror("Error", f"Executable {exe} not found! Run the bash script first."))
            self.root.after(0, self.reset_run_btn)
            return

        try:
            if self.stop_requested: return
                
            self.root.after(0, lambda: self.lbl_status.config(text=f"Status: Executing {exe} ..."))
            self.root.after(0, lambda: self.append_console(f"\n>> STARTING {exe} <<\n"))
            
            # Start the C++ program and capture its output line by line in real-time
            self.current_process = subprocess.Popen([exe, img_path, out_dir, "3"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            current_res, current_s, current_p = "", "", ""
            
            # Read standard output iteratively to update the UI without waiting for completion
            for line in self.current_process.stdout:
                if self.stop_requested:
                    break
                    
                self.root.after(0, self.append_console, line)
                
                # Parse the console output to detect when a new workload starts processing
                match_start = re.search(r'\[(\d+)x\d+\] Workload \(Search: (\d+),\s*Patch: (\d+)\)', line)
                if match_start:
                    current_res, current_s, current_p = match_start.group(1), match_start.group(2), match_start.group(3)
                    self.root.after(0, self.update_live_images_during_run, current_res, current_s, current_p, out_dir)
                    
                # Update images right after CPU computation finishes
                if "[CPU] Average:" in line and current_res:
                    self.root.after(0, self.update_live_images_during_run, current_res, current_s, current_p, out_dir)
                    
                # Update images right after GPU computation finds the best block size
                if "Best block size:" in line and current_res:
                    self.root.after(0, self.update_live_images_during_run, current_res, current_s, current_p, out_dir)

            self.current_process.wait()

            if self.stop_requested:
                return

            # Raise an error if the process crashed, unless the user intentionally stopped it
            if self.current_process.returncode != 0 and not self.stop_requested:
                raise subprocess.CalledProcessError(self.current_process.returncode, exe)

            self.root.after(0, self.benchmark_success)
            
        except subprocess.CalledProcessError as e:
            if not self.stop_requested:
                self.root.after(0, lambda e=e: messagebox.showerror("Execution Error", f"Failed to run executable:\n{e}"))
                self.root.after(0, self.reset_run_btn)

    def benchmark_success(self):
        # Update UI indicators when tests finish normally
        self.lbl_status.config(text="Status: Benchmark Completed Successfully!", foreground="green")
        self.append_console("\n>> ALL DONE! Results are ready in Evaluate Tab. <<\n")
        self.reset_run_btn()
        messagebox.showinfo("Success", "Tests finished! Check 'Evaluate Results' tab for graphs.")
        
        # Scan the folder to load the generated files into the viewer
        self.scan_output_folder(self.current_out_dir)

    def reset_run_btn(self):
        # Restore normal button states after run completion or cancellation
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")

    def update_live_images_during_run(self, res, search, patch, out_dir):
        # Display the active workload parameters in the viewer title
        self.lbl_current_view.config(text=f"Live View -> Res: {res}x{res} | Search: {search} | Patch: {patch}", foreground="navy")

        # Locate the specific files corresponding to the current resolution and workload
        orig_file, cpu_file, gpu_file = self.get_image_paths(out_dir, res, search, patch)

        self.image_references.clear()
        
        # Load and resize the found images into the canvas labels
        self.load_and_set_image(orig_file, self.lbl_orig_img, 400, f"Original image not found for {res}x{res}")
        self.load_and_set_image(cpu_file, self.lbl_cpu_img, 500, f"CPU filtered image not found for S:{search} P:{patch}")
        self.load_and_set_image(gpu_file, self.lbl_gpu_img, 500, f"GPU filtered image not found for S:{search} P:{patch}")

    def load_folder_directly(self):
        # Allow users to analyze results from a previously completed benchmark run
        dirname = filedialog.askdirectory(title="Select a specific output folder")
        if dirname:
            self.current_out_dir = dirname
            self.scan_output_folder(dirname)

    def find_existing_image(self, patterns):
        # Iterate over possible file naming patterns and return the first valid match found
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
        return None

    def get_image_paths(self, folder_path, res, search, patch):
        # Clean input strings just in case
        res = str(res).strip()
        search = str(search).strip()
        patch = str(patch).strip()

        # Define fallback naming patterns for original images to ensure compatibility
        original_patterns = [
            os.path.join(folder_path, f"noisy_{res}.png"),
            os.path.join(folder_path, f"original_{res}.png"),
            os.path.join(folder_path, f"resized_{res}.png"),
            os.path.join(folder_path, f"input_{res}.png"),
            os.path.join(folder_path, f"*{res}*original*.png"),
            os.path.join(folder_path, f"*{res}*noisy*.png"),
        ]

        # Define fallback naming patterns for CPU results
        cpu_patterns = [
            os.path.join(folder_path, f"denoised_cpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"filtered_cpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"cpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"*cpu*{res}*s{search}*p{patch}*.png"),
            os.path.join(folder_path, f"*{res}*s{search}*p{patch}*cpu*.png"),
        ]

        # Define fallback naming patterns for GPU results
        gpu_patterns = [
            os.path.join(folder_path, f"denoised_gpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"filtered_gpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"gpu_{res}_s{search}_p{patch}.png"),
            os.path.join(folder_path, f"*gpu*{res}*s{search}*p{patch}*.png"),
            os.path.join(folder_path, f"*{res}*s{search}*p{patch}*gpu*.png"),
        ]

        return (
            self.find_existing_image(original_patterns),
            self.find_existing_image(cpu_patterns),
            self.find_existing_image(gpu_patterns),
        )

    def scan_output_folder(self, folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Selected output folder is not valid.")
            return

        # Gather all files that look like CPU output to build a list of tested configurations
        files = []
        for pattern in ["denoised_cpu_*.png", "filtered_cpu_*.png", "cpu_*.png", "*cpu*.png"]:
            files.extend(glob.glob(os.path.join(folder_path, pattern)))
        files = sorted(set(files))

        json_path = os.path.join(folder_path, "results.json")
        best_configs = {}

        # If a JSON log exists, read it to identify the configurations with the lowest error (MSE)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for d in data:
                        res = str(d.get("resolution", "")).strip()
                        if not res:
                            continue
                        
                        mse = d.get("mse", float('inf'))
                        
                        # Store the workload parameters that yielded the minimum error for each resolution
                        if res not in best_configs or mse < best_configs[res]["mse"]:
                            best_configs[res] = {
                                "mse": mse,
                                "search": str(d.get("search_radius", "")).strip(),
                                "patch": str(d.get("patch_radius", "")).strip()
                            }
            except Exception as e:
                print(f"JSON Parse Error: {e}")

        configs = set()
        
        # Regex patterns to extract resolution, search radius, and patch radius from filenames
        file_regex = re.compile(r'(?:denoised_|filtered_)?cpu_(\d+)_s(\d+)_p(\d+)\.png$', re.IGNORECASE)
        loose_regex = re.compile(r'(\d+).*s(\d+).*p(\d+).*cpu|cpu.*(\d+).*s(\d+).*p(\d+)', re.IGNORECASE)

        # Parse every found image name and format it for the combobox dropdown
        for file_path in files:
            basename = os.path.basename(file_path)
            match = file_regex.search(basename)
            if match:
                res, search, patch = match.group(1), match.group(2), match.group(3)
                configs.add(f"Res: {res}x{res} | Search: {search} | Patch: {patch}")
                continue

            loose = loose_regex.search(basename)
            if loose:
                groups = [g for g in loose.groups() if g]
                if len(groups) >= 3:
                    res, search, patch = groups[0], groups[1], groups[2]
                    configs.add(f"Res: {res}x{res} | Search: {search} | Patch: {patch}")

        # Sort the dropdown options logically by resolution, then search radius, then patch radius
        configs = sorted(
            configs,
            key=lambda x: (
                int(x.split("x")[0].split(":")[1].strip()),
                int(x.split("Search:")[1].split("|")[0].strip()),
                int(x.split("Patch:")[1].strip())
            )
        )

        final_list = []
        
        # Inject the best performing configurations at the top of the list for quick access
        if best_configs:
            for res in ["256", "512", "1024", "2048", "4096"]:
                if res in best_configs:
                    s = best_configs[res]["search"]
                    p = best_configs[res]["patch"]
                    if s and p:
                        final_list.append(f" BEST OF {res}x{res} (Top Quality) | Search: {s} | Patch: {p}")

            # Add a visual separator if standard configurations follow
            if final_list and configs:
                final_list.append("---------------------------------------------------------------")

        final_list.extend(configs)

        # Update the UI dropdown with the final parsed list
        if final_list:
            self.combo_config['values'] = final_list
            first_real_index = 0
            if final_list[0].startswith("---") and len(final_list) > 1:
                first_real_index = 1
            self.combo_config.current(first_real_index)
            self.update_image_display()
        else:
            self.combo_config['values'] = []
            self.lbl_orig_img.config(image='', text="No original image found")
            self.lbl_cpu_img.config(image='', text="No CPU filtered image found")
            self.lbl_gpu_img.config(image='', text="No GPU filtered image found")
            self.image_references.clear()
            messagebox.showinfo("Info", "No generated images found in the output directory.")

    def update_image_display(self, event=None):
        selection = self.config_var.get()
        if not selection or selection.startswith("---"):
            return

        # Extract numerical parameters from the formatted dropdown text based on its structure
        try:
            if selection.startswith(" BEST"):
                self.lbl_current_view.config(text=selection, foreground="goldenrod")
                parts = selection.split("|")
                res = parts[0].split("OF")[1].split("(")[0].strip().split("x")[0]
                search = parts[1].split(":")[1].strip()
                patch = parts[2].split(":")[1].strip()
            else:
                self.lbl_current_view.config(text=selection, foreground="navy")
                parts = selection.split("|")
                res = parts[0].split(":")[1].split("x")[0].strip()
                search = parts[1].split(":")[1].strip()
                patch = parts[2].split(":")[1].strip()
        except Exception as e:
            messagebox.showerror("Parse Error", f"Selected result could not be read:\n{selection}\n\n{e}")
            return

        orig_file, cpu_file, gpu_file = self.get_image_paths(self.current_out_dir, res, search, patch)

        self.image_references.clear()
        self.lbl_gpu_title.config(text="Filtered (GPU)")

        self.load_and_set_image(orig_file, self.lbl_orig_img, 400, f"Original image not found for {res}x{res}")
        self.load_and_set_image(cpu_file, self.lbl_cpu_img, 500, f"CPU filtered image not found for S:{search} P:{patch}")
        self.load_and_set_image(gpu_file, self.lbl_gpu_img, 500, f"GPU filtered image not found for S:{search} P:{patch}")

    def load_and_set_image(self, path, label_widget, max_size, missing_text="Image not found"):
        if path and os.path.exists(path):
            try:
                # Open image and apply high-quality Lanczos downsampling to fit UI boundaries
                img = Image.open(path)
                img.thumbnail((max_size, max_size), Image.LANCZOS)
                
                # Convert to Tkinter format and save to list to avoid garbage collection removal
                tk_img = ImageTk.PhotoImage(img)
                self.image_references.append(tk_img)
                label_widget.config(image=tk_img, text="")
            except Exception as e:
                label_widget.config(image='', text=f"Image load error:\n{os.path.basename(path)}\n{e}")
        else:
            label_widget.config(image='', text=missing_text)

    def setup_eval_tab(self):
        # Build the top section for JSON file selection and execution
        top_frame = ttk.Frame(self.tab_eval)
        top_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(top_frame, text="Select JSON Results File:").pack(side='left', padx=5)
        self.json_path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.json_path_var, width=60).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Browse JSON", command=self.browse_json).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Generate Graphs & Table", command=self.load_results).pack(side='left', padx=15)

        # Setup vertical split pane to show table on top and graphs at the bottom
        self.paned = ttk.PanedWindow(self.tab_eval, orient=tk.VERTICAL)
        self.paned.pack(fill='both', expand=True, padx=10, pady=10)

        # Build the structured data table using Treeview
        table_frame = ttk.LabelFrame(self.paned, text="Benchmark Data Table")
        self.paned.add(table_frame, weight=1)
        
        cols = ("Res", "Search", "Patch", "CPU Time (s)", "GPU Time (s)", "Speedup", "Best Block", "MSE")
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=6)
        
        # Apply column definitions dynamically
        for c in cols: self.tree.heading(c, text=c)
        for c in cols: self.tree.column(c, width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)

        # Build the dashboard container to hold the matplotlib figures
        self.graph_outer_frame = ttk.LabelFrame(self.paned, text="Performance Analysis Dashboards")
        self.paned.add(self.graph_outer_frame, weight=5)

        self.graph_canvas = tk.Canvas(self.graph_outer_frame)
        self.graph_scrollbar = ttk.Scrollbar(self.graph_outer_frame, orient="vertical", command=self.graph_canvas.yview)
        
        self.graph_inner_frame = ttk.Frame(self.graph_canvas)
        self.graph_inner_frame.bind("<Configure>", lambda e: self.graph_canvas.configure(scrollregion=self.graph_canvas.bbox("all")))
        self.graph_canvas_window = self.graph_canvas.create_window((0, 0), window=self.graph_inner_frame, anchor="nw")
        
        self.graph_canvas.bind("<Configure>", lambda e: self.graph_canvas.itemconfig(self.graph_canvas_window, width=e.width))
        self.graph_canvas.configure(yscrollcommand=self.graph_scrollbar.set)
        
        self.graph_scrollbar.pack(side="right", fill="y")
        self.graph_canvas.pack(side="left", fill="both", expand=True)
        
        self._bind_mouse_scroll(self.graph_canvas, self.graph_canvas)

    def browse_json(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename: self.json_path_var.set(filename)

    def load_results(self):
        # Verify JSON path before processing
        path = self.json_path_var.get()
        if not os.path.exists(path): return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read JSON: {e}")
            return

        # Clear existing table rows before inserting new data
        for item in self.tree.get_children(): self.tree.delete(item)

        # Extract and format numerical fields from the JSON objects into table rows
        for row in data:
            cpu_t = row.get("cpu_avg_s", 0)
            gpu_t = row.get("gpu_avg_s", 0)
            speedup = row.get("speedup", 0)
            block = row.get("best_block_size", "-")
            mse = row.get("mse", 0)
            
            self.tree.insert("", "end", values=(
                f"{row.get('resolution')}x{row.get('resolution')}", 
                row.get("search_radius"), 
                row.get("patch_radius"), 
                f"{cpu_t:.4f}", 
                f"{gpu_t:.4f}", 
                f"{speedup:.2f}x" if speedup > 0 else "-", 
                block,
                f"{mse:.5f}" if "mse" in row else "-"
            ))

        # Render visual charts using the parsed data
        self.draw_graphs(data)

    def draw_graphs(self, data):
        # Destroy old graphs to prevent memory leaks and overlapping plots
        for widget in self.graph_inner_frame.winfo_children(): 
            widget.destroy()

        # Create a 2x2 grid for matplotlib subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Group speedup values logically by specific workload profiles
        workloads = {}
        for d in data:
            if "speedup" in d and "search_radius" in d and "patch_radius" in d:
                w_key = (d["search_radius"], d["patch_radius"])
                if w_key not in workloads:
                    workloads[w_key] = {'res': [], 'speedup': []}
                workloads[w_key]['res'].append(str(d["resolution"]))
                workloads[w_key]['speedup'].append(d["speedup"])

        colors = ['green', 'purple', 'orange', 'brown', 'magenta', 'cyan']
        color_idx = 0
        all_speedups = []
        points_by_x = {}

        # Plot 1: GPU Speedup scaling over different resolutions
        if workloads:
            for w_key in sorted(workloads.keys()):
                s, p = w_key
                res_vals = workloads[w_key]['res']
                sp_vals = workloads[w_key]['speedup']
                
                color_to_use = colors[color_idx % len(colors)]
                axs[0, 0].plot(res_vals, sp_vals, marker='o', linewidth=2, markersize=6, 
                               label=f'Search={s}, Patch={p}', color=color_to_use)
                
                # Store coordinates to render text annotations later without overlap
                for x_val, y_val in zip(res_vals, sp_vals):
                    if x_val not in points_by_x:
                        points_by_x[x_val] = []
                    points_by_x[x_val].append((y_val, color_to_use, f"{y_val:.1f}x"))
                
                all_speedups.extend(sp_vals)
                color_idx += 1
                
            axs[0, 0].set_title('GPU Speedup vs Resolution (All Workloads)', fontsize=11, fontweight='bold')
            axs[0, 0].set_xlabel('Resolution')
            axs[0, 0].set_ylabel('Speedup Factor (x)')
            axs[0, 0].grid(True, linestyle='--', alpha=0.6)
            axs[0, 0].legend(fontsize=9, loc='upper left')

            if all_speedups:
                min_y, max_y = min(all_speedups), max(all_speedups)
                axs[0, 0].set_ylim(min_y * 0.80, max_y * 1.30)
                
            # Apply offset logic to text labels so they remain readable
            for x_val, pts in points_by_x.items():
                pts.sort(key=lambda x: x[0], reverse=True)
                for i, (y_val, color, txt) in enumerate(pts):
                    x_offset = 12 if i % 2 == 0 else -12
                    ha_align = 'left' if i % 2 == 0 else 'right'
                    y_offset = (len(pts) // 2 - i) * 8 

                    axs[0, 0].annotate(txt, xy=(x_val, y_val), xytext=(x_offset, y_offset), textcoords="offset points", color=color, fontsize=9, fontweight='bold', ha=ha_align, va='center')
        else:
            axs[0, 0].text(0.5, 0.5, 'No Speedup Data Available', ha='center')

        # Plot 2: Bar chart comparing block size efficiency
        block_labels, block_times = [], []
        for d in data:
            if d.get("resolution") == 1024 and "block_size_sweep" in d:
                for b in d["block_size_sweep"]:
                    block_labels.append(b["block"])
                    block_times.append(b["avg_s"])
                break
        
        if block_labels:
            bars = axs[0, 1].bar(block_labels, block_times, color='cornflowerblue', edgecolor='black')
            axs[0, 1].set_title('GPU Time vs Block Size (Res: 1024)', fontsize=11, fontweight='bold')
            axs[0, 1].set_xlabel('Thread Block Size')
            axs[0, 1].set_ylabel('Execution Time (s)')
            
            max_t = max(block_times) if block_times else 0.1
            axs[0, 1].set_ylim(0, max_t * 1.25)
            
            # Place the exact seconds value clearly above each bar
            for bar in bars:
                yval = bar.get_height()
                axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + (max_t * 0.03), 
                               f"{yval:.4f}s", ha='center', va='bottom', fontsize=9)
        else:
            axs[0, 1].text(0.5, 0.5, 'No Block Sweep Data', ha='center')

        # Plot 3: Line graph depicting performance degradation as search radius increases
        s_rads, s_cpu, s_gpu = [], [], []
        for d in data:
            if d.get("resolution") == 1024:
                s_rads.append(str(d["search_radius"]))
                s_cpu.append(d.get("cpu_avg_s", 0))
                s_gpu.append(d.get("gpu_avg_s", 0))
                
        if s_rads:
            axs[1, 0].plot(s_rads, s_cpu, marker='s', color='red', label='CPU Time')
            if any(v > 0 for v in s_gpu):
                axs[1, 0].plot(s_rads, s_gpu, marker='^', color='blue', label='GPU Time')
            axs[1, 0].set_yscale('log') 
            axs[1, 0].set_title('Time vs Search Radius (Res: 1024)', fontsize=11, fontweight='bold')
            axs[1, 0].set_xlabel('Search Radius')
            axs[1, 0].set_ylabel('Time (s) [Log Scale]')
            axs[1, 0].legend()
            axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)

        # Plot 4: Hardware comparison across distinct workload definitions
        p_rads, p_cpu, p_gpu = [], [], []
        for d in data:
            if d.get("resolution") == 512:
                label = f"S:{d['search_radius']}/P:{d['patch_radius']}"
                p_rads.append(label)
                p_cpu.append(d.get("cpu_avg_s", 0))
                p_gpu.append(d.get("gpu_avg_s", 0))

        if p_rads:
            axs[1, 1].plot(p_rads, p_cpu, marker='o', color='firebrick', label='CPU Time')
            if any(v > 0 for v in p_gpu):
                axs[1, 1].plot(p_rads, p_gpu, marker='o', color='navy', label='GPU Time')
            axs[1, 1].set_yscale('log')
            axs[1, 1].set_title('Time vs Workload (Search/Patch) (Res: 512)', fontsize=11, fontweight='bold')
            axs[1, 1].set_xlabel('Workload Profile')
            axs[1, 1].set_ylabel('Time (s) [Log Scale]')
            axs[1, 1].legend()
            axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)
            # Rotate axis labels slightly to avoid text overlap
            plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=15)

        # Apply layout padding to prevent titles from merging with axes
        fig.tight_layout(pad=3.0, h_pad=8.0, w_pad=3.0)

        # Render the complete matplotlib figure into the Tkinter application layer
        canvas = FigureCanvasTkAgg(fig, master=self.graph_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = NLMBenchmarkApp(root)
    root.mainloop()