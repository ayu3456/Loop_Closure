# ORB-SLAM3 Python Implementation

A Python implementation of ORB-SLAM3 with real-time visualization. This project provides a lightweight version of the SLAM system with focus on monocular tracking and visualization.

## Features

- Real-time monocular SLAM
- ORB feature detection and tracking
- Scale-consistent motion estimation
- Real-time 3D trajectory visualization
- Support for video files and image sequences
- Camera pose estimation with essential matrix 
decomposition
- Interactive 3D visualization with camera frustum

## Project Structure

```
orbslam_in_python/
├── config/
│   └── camera_config.yaml    # Camera calibration parameters
├── data/
│   └── rgbd_dataset_freiburg1_xyz/  # Example dataset
├── src/
│   ├── system.py            # Main SLAM system
│   ├── tracking.py          # Feature tracking and pose estimation
│   ├── mapping.py           # Mapping module (basic implementation)
│   └── run_slam.py          # Main script to run the system
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/orbslam_in_python.git
cd orbslam_in_python
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

### Running with Image Sequence

```bash
python src/run_slam.py --config config/camera_config.yaml --input path/to/image/sequence --viz_delay 0.05
```

### Running with Video File

```bash
python src/run_slam.py --config config/camera_config.yaml --input path/to/video.mp4 --viz_delay 0.05
```

### Running with Webcam

```bash
python src/run_slam.py --config config/camera_config.yaml --input 0 --viz_delay 0.05
```

### Command Line Arguments

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

- OpenCV (cv2)
- NumPy
- Matplotlib
- PyYAML

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
- No loop closure
- Scale drift may occur in long sequences

## Contributing

Feel free to open issues or submit pull requests for improvements. Some areas that could be enhanced:

- Loop closure detection
- Local bundle adjustment
- Keyframe management
- Map point culling
- Multi-threading support

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

This implementation is inspired by the original ORB-SLAM3 paper:
"ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM" 
