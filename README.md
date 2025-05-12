## Edge Detection with Auto-Canny and Temporal Smoothing

This repository implements an edge detection pipeline using OpenCV, featuring:

* **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing
* **Gaussian Blur** smoothing
* **Auto-Canny** edge detection with dynamic thresholding based on image median
* **Temporal smoothing** for video streams using exponential moving average
* **Motion detection** to augment edges with moving object highlights
* **Evaluation** of Precision, Recall, and F1-score against ground truth skeletons (using distance transform)

### Repository Structure

```
|── C++\ code/ 
    ├── build/                  # Compiled binaries and object files
    │   └── edge_detection      # Edge detection executable
    ├── Experiments/            # Source code
    │   ├── BSDS_tuned.cpp      # Using mean instead of median for adaptive thresholding
    │   ├── clahe_improv.cpp    # CLAHE & Gaussian blur logic
    │   ├── edge_detection.cpp  # Self implementation of Canny
    │   ├── flicker_motion.cpp  # Temporal smoothening Operations
    |── detect_main.cpp         # The main code for edge detection, derived from the above experiments
    ├── rebuild.sh              # Clean and rebuild script
    ├── convert_mat_jpg.py.sh   # Helper code to preprocess BSDS500 dataset
    ├── plots.py                # Visualise the numberical results
└── README.md               # This file
```

---

## Prerequisites

* **C++17** (or later) compiler (e.g., **g++** 9+)
* **CMake** 3.10+
* **OpenCV** 4.x installed and discoverable via CMake
* **Filesystem** library (C++17 standard)

On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

Download the BSDS500 dataset for evaluation of the pipeline. The ground truth in the original dataset consists of `.mat` files. Convert them into jpg to work easily with C++ using the `convert_mat_jpg.py` script.
```
├── archive/                # Dataset folders
│   ├── images/             # Input images for train/test splits
│   │   ├── train/
│   │   └── test/
│   └── converted_ground_truth/  # Ground truth skeleton masks
│       ├── train/
│       └── test/
```
---

## Building the Project

A helper script is provided to clean and rebuild the project:

```bash
# Make sure script is executable
chmod +x rebuild.sh

# Run the rebuild script
./rebuild.sh

# The executable will be generated at:
  ./build/edge_detection
```

`rebuild.sh` performs:

1. Remove `build/` directory
2. Create fresh `build/`
3. Run `cmake ..` and `make`

---

## Usage

Run the edge detection executable with the following options:

```bash
# Evaluate on image splits (train)
./build/edge_detection --train
# Computes Precision, Recall, F1 and logs results in csv:
#   bsds_train_autoCanny_<sigma>.csv

# Evaluate on train split explicitly using --eval (alias for --train)
./build/edge_detection --eval train

# Process a video file with temporal smoothing (--eval mode)
./build/edge_detection --eval path/to/video.mp4
```

### Command-Line Parameters

| Flag             | Description                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `--train`        | Run evaluation on the `archive/images/train` split. Generates `bsds_train_autoCanny_<sigma>.csv` |
| `--eval <VIDEO>` | Run video edge detection on specified file with temporal smoothing and motion detection. If no video file path is specified it defaults to the camera.       |
| *(no args)*      | Returns `-1` and prints usage instructions.                                                      |

*sigma* refers to the Auto-Canny sigma value configured in source (default `0.33`).

---

## Output

* **Image Evaluation**: CSV logs with columns: `image, P, R, F1, ms`

  * Stored as `bsds_<split>_autoCanny_<sigma>.csv`

* **Video Processing**: Real-time GUI window showing `Original | AutoEdges` side by side. Press `q` to quit.

---

## Algorithm Details

1. **Preprocessing**

   * Apply CLAHE (clip limit 2.0, tile size 8x8)
   * Gaussian blur (5x5, σ=1.5)
2. **Auto-Canny**

   * Compute median of pixel intensities
   * Set thresholds: `low = max(0, (1 - σ) * median)`, `high = min(255, (1 + σ) * median)`
   * Run Canny + dilate (2x2 rectangular kernel)
3. **Temporal Smoothing**

   * Exponential moving average on edge maps (α = 0.25)
4. **Motion Augmentation**

   * Frame differencing with threshold 30 to detect motion
   * Combine motion mask with edges via bitwise OR
5. **Evaluation (PRF)**

   * Skeletonize GT via dilation and OR across annotation images
   * Compute distance transforms on inverted GT and DET
   * Count TP, FP, FN within radius 5 pixels
   * Calculate Precision, Recall, F1-score

---

