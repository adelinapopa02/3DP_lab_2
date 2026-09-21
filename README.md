# 3D Data Processing – Lab 2: Structure from Motion

This project implements a basic **Structure from Motion (SfM)** pipeline in C++: feature extraction/matching across an image sequence, followed by **bundle-adjustment-based 3D reconstruction** (via Ceres) to recover camera poses and a sparse point cloud.

The pipeline:

1. **`matcher`** — detects and matches keypoints across the image sequence (classical features, or **modern features** via an external SuperPoint extractor), using the camera calibration to filter matches, and writes the correspondences to a data file.
2. **`basic_sfm`** — consumes that data file and solves the SfM optimization problem, exporting the reconstructed **point cloud (`.ply`)**.

---

## Author

* [@adelinapopa02](https://github.com/adelinapopa02)

---

## Project overview

| File | Role |
|---|---|
| `features_matcher.h` / `.cpp` | Keypoint detection + descriptor matching across the image sequence. |
| `basic_sfm.h` / `.cpp` | SfM optimization (camera poses + 3D points) via Ceres, robust (Huber) or plain (null) loss. |
| `io_utils.h` / `.cpp` | Calibration/image-list loading and data file I/O. |
| `matcher_app.cpp` | CLI entry point for the feature matching stage. |
| `sfm_app.cpp` | CLI entry point for the reconstruction stage. |
| `modern_features/extract_superpoint.py` | Optional SuperPoint-based feature extractor used when the "modern features" flag is enabled. |
| `point_clouds/` | Example reconstructions for both provided datasets, with ORB/modern features × Huber/null loss combinations. |

---

## Prerequisites

```bash
sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev libomp-dev
sudo apt install libceres-dev libyaml-cpp-dev libgtest-dev libeigen3-dev
```

(not required inside the course virtual machine, where these are already installed)

## Build

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Executables are produced in `build/bin/`.

## Run

```bash
./matcher <calibration file> <images folder> <output data file> <use modern features (0|1)> [focal length scale]
./basic_sfm <input data file> <output ply file>
```

### Example (provided datasets, focal length scale 1.1)

```bash
./matcher ../datasets/3dp_cam.yml ../datasets/images_1 data1.txt 0 1.1
./matcher ../datasets/3dp_cam.yml ../datasets/images_2 data2.txt 0 1.1

./basic_sfm data1.txt cloud1.ply
./basic_sfm data2.txt cloud2.ply
```

## Datasets

The `dataset/` folder contains two image sets with their calibration files. Preprocessed data files (already matched features) are provided for convenience, but final evaluation uses the raw input images, not the preprocessed ones.
