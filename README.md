# ORB-SLAM3 Python Implementation

A Python implementation of ORB-SLAM3 with real-time visualization. This project provides a lightweight version of the SLAM system with focus on monocular tracking and visualization.

## Features

- Real-time monocular SLAM
- ORB feature detection and tracking
- Scale-consistent motion estimation
- Real-time 3D trajectory visualization
- Support for video files and image sequences
- Camera pose estimation with essential matrix decomposition
- Interactive 3D visualization with camera frustum
- CNN-based Loop Closure Detection (integrated into SLAM, also runnable standalone)

## Project Structure

```
Loop_Closure/                   # Project Root
├── config/
│   └── camera_config.yaml            # Camera calibration parameters
├── data/                           # (Example dataset directory, if used)
│   └── rgbd_dataset_freiburg1_xyz/
├── src/
│   ├── system.py                     # Main SLAM system with integrated loop closure
│   ├── tracking.py                   # Feature tracking and pose estimation
│   ├── mapping.py                    # Mapping module (basic implementation)
│   └── run_slam.py                   # Main script to run the SLAM system
├── cnn_loop_closure.py               # CNN Loop Closure detection module
├── run_cnn_loop_closure_on_video.py  # Script to run standalone CNN loop closure
├── cnn_rf_loop_closure_model.joblib  # Pre-trained Random Forest model for loop closure
├── cnn_rf_scaler.joblib              # Pre-trained scaler for loop closure features
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Specifies intentionally untracked files
├── orb_slam_keyframes/               # Directory for saved keyframes (auto-generated, in .gitignore)
└── output/                           # Directory for script outputs (e.g., trajectories, auto-generated, in .gitignore)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayu3456/Loop_Closure.git
cd Loop_Closure
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Integrated ORB-SLAM3 System (with CNN Loop Closure)

The main SLAM system now incorporates CNN-based loop closure detection, which runs periodically.

#### Running with Image Sequence
```bash
python src/run_slam.py --config config/camera_config.yaml --input path/to/image/sequence --viz_delay 0.05
```

#### Running with Video File
```bash
python src/run_slam.py --config config/camera_config.yaml --input path/to/video.mp4 --viz_delay 0.05
```

#### Running with Webcam
```bash
python src/run_slam.py --config config/camera_config.yaml --input 0 --viz_delay 0.05
```

### Running Standalone CNN Loop Closure Detection

You can also run the CNN loop closure detection module independently on a directory of frames or a video file. This is useful for testing the loop closure component or processing pre-recorded data.

#### Using a Directory of Pre-extracted Frames:
```bash
python run_cnn_loop_closure_on_video.py --frames_dir path/to/your/frames_directory --output_dir path/to/output_results_directory
```
- Ensure frames are named sequentially (e.g., `frame_0001.png`, `frame_0002.jpg`, etc.).
- The script will generate a 3D trajectory visualization (`cnn_loop_closure_3d_trajectory.html`) and other results in the specified output directory.

#### Using a Video File:
```bash
python run_cnn_loop_closure_on_video.py --video_file path/to/your/video.mp4 --output_dir path/to/output_results_directory
```
- The script will first extract frames from the video into a subdirectory within the output directory.

### Command Line Arguments for `src/run_slam.py`

- `--config`: Path to camera configuration file (YAML)
- `--input`: Path to input source (video file, image directory, or camera index)
- `--output`: Optional path to save trajectory (as .npy file)
- `--viz_delay`: Delay between frames for visualization (seconds)

## Example Dataset

The system has been tested with the TUM RGB-D dataset (freiburg1_xyz sequence). You can download it from:
https://vision.in.tum.de/data/datasets/rgbd-dataset/download

## Visualization

The system provides two visualization windows:

1. **Feature Visualization** (OpenCV window)
   - Green dots: Detected ORB features
   - Coordinate axes: Current camera orientation
   - Press 'q' to quit

2. **3D Trajectory** (Matplotlib window)
   - Blue line: Camera trajectory
   - Red dot: Current camera position
   - Green frustum: Camera orientation
   - Interactive 3D view (rotate, zoom)
  
<img width="1280" alt="Screenshot 2025-05-20 at 8 13 28 PM" src="https://github.com/user-attachments/assets/0f15a04e-cbfe-4306-bd3c-f7d6d189e36c" />

## Dependencies

The project relies on several Python libraries. Key dependencies include:

- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- PyYAML (`PyYAML`)
- PyTorch (`torch`) & Torchvision (`torchvision`) for CNN features
- Scikit-learn (`scikit-learn`) for the Random Forest classifier
- Joblib (`joblib`) for loading pre-trained models
- Plotly (`plotly`) for interactive 3D visualizations
- Pillow (`Pillow`) for image manipulation
- Requests (`requests`)
- TQDM (`tqdm`) for progress bars

All dependencies are listed in `requirements.txt` and can be installed via:
```bash
pip install -r requirements.txt
```

## Configuration

The camera parameters can be configured in `config/camera_config.yaml`:

```yaml
Camera:
  # Camera matrix parameters
  fx: 517.306408
  fy: 516.469215
  cx: 318.643040
  cy: 255.313989

  # Distortion coefficients
  k1: 0.262383
  k2: -0.953104
  p1: -0.005358
  p2: 0.002628
  k3: 1.163314

  # Image dimensions
  width: 640
  height: 480
```

## Limitations

- Monocular-only implementation (no stereo or RGB-D support)
- Basic mapping functionality
- Loop closure detection is implemented and identifies potential loops, but full pose graph optimization and map correction based on these loops are not yet integrated.
- Scale drift may occur in long sequences (inherent in monocular SLAM without loop closure correction or other scale-aware sensors).

## Contributing

Feel free to open issues or submit pull requests for improvements. Some areas that could be enhanced:

- Pose graph optimization and map correction using detected loop closures
- Robust loop verification mechanisms
- Local bundle adjustment
- Keyframe management strategies
- Map point culling
- Multi-threading support for performance improvement

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

This implementation is inspired by the original ORB-SLAM3 paper:
"ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM" 
