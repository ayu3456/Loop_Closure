import sys
import os

# Add project root to sys.path to allow absolute imports from root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import yaml
import argparse
from system import ORBSlam3
from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()  # Enable interactive mode for real-time plotting

def load_config(config_path: str) -> dict:
    """Load camera configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_camera_matrix(config: dict) -> tuple:
    """Create camera matrix and distortion coefficients from config."""
    camera_config = config['Camera']
    
    # Create camera matrix
    camera_matrix = np.array([
        [camera_config['fx'], 0, camera_config['cx']],
        [0, camera_config['fy'], camera_config['cy']],
        [0, 0, 1]
    ])
    
    # Create distortion coefficients
    dist_coeffs = np.array([
        camera_config['k1'],
        camera_config['k2'],
        camera_config['p1'],
        camera_config['p2'],
        camera_config['k3']
    ])
    
    return camera_matrix, dist_coeffs

class TrajectoryVisualizer:
    def __init__(self):
        """Initialize the 3D trajectory visualizer."""
        # Create figure and 3D axes
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Camera Trajectory')
        
        # Initialize trajectory storage
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        
        # Set initial view limits with margin
        self.margin = 0.5
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        
        # Initialize plot objects
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', label='Camera Path', linewidth=2)
        self.current_pos = self.ax.scatter([], [], [], c='r', marker='o', s=100, label='Current Position')
        
        # Camera frustum visualization
        self.camera_frustum = []
        self.frustum_size = 0.15
        self.frustum_color = 'g'
        self.frustum_alpha = 0.3
        
        # Visualization parameters
        self.update_interval = 3  # Update every N frames
        self.frame_count = 0
        self.smooth_factor = 0.3  # For smooth view limit updates
        self.min_view_range = 1.0  # Minimum view range
        
        # Initialize view
        self.ax.view_init(elev=30, azim=45)
        self.ax.grid(True)
        self.ax.legend()
        
        # Enable tight layout and interactive mode
        self.fig.tight_layout()
        plt.ion()
        
    def update(self, pose: np.ndarray):
        """
        Update the trajectory visualization with new pose.
        
        Args:
            pose: 4x4 camera pose matrix
        """
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return
            
        # Extract position from pose matrix
        position = pose[:3, 3]
        
        # Update trajectory
        self.trajectory_x.append(position[0])
        self.trajectory_y.append(position[1])
        self.trajectory_z.append(position[2])
        
        # Update trajectory line
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
        self.trajectory_line.set_3d_properties(self.trajectory_z)
        
        # Update current position
        self.current_pos._offsets3d = ([position[0]], [position[1]], [position[2]])
        
        # Update camera frustum
        self._update_camera_frustum(pose)
        
        # Smoothly adjust view limits
        self._update_view_limits()
        
        # Refresh display
        self.fig.canvas.draw()
        plt.pause(0.001)
        
    def _update_camera_frustum(self, pose: np.ndarray):
        """
        Update the camera frustum visualization.
        
        Args:
            pose: 4x4 camera pose matrix
        """
        # Remove previous frustum
        for line in self.camera_frustum:
            if line is not None:
                line.remove()
        self.camera_frustum = []
        
        # Define frustum points in camera coordinates
        frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [-self.frustum_size, -self.frustum_size, self.frustum_size * 2],  # Bottom-left
            [self.frustum_size, -self.frustum_size, self.frustum_size * 2],   # Bottom-right
            [self.frustum_size, self.frustum_size, self.frustum_size * 2],    # Top-right
            [-self.frustum_size, self.frustum_size, self.frustum_size * 2],   # Top-left
        ])
        
        # Transform points to world coordinates
        world_points = (pose[:3, :3] @ frustum_points.T + pose[:3, 3:]).T
        
        # Draw frustum lines
        center = world_points[0]
        for i in range(1, 5):
            # Draw lines from center to corners
            line = self.ax.plot([center[0], world_points[i][0]],
                              [center[1], world_points[i][1]],
                              [center[2], world_points[i][2]],
                              color=self.frustum_color,
                              alpha=self.frustum_alpha,
                              linewidth=1)[0]
            self.camera_frustum.append(line)
            
            # Draw lines between corners
            next_i = 1 if i == 4 else i + 1
            line = self.ax.plot([world_points[i][0], world_points[next_i][0]],
                              [world_points[i][1], world_points[next_i][1]],
                              [world_points[i][2], world_points[next_i][2]],
                              color=self.frustum_color,
                              alpha=self.frustum_alpha,
                              linewidth=1)[0]
            self.camera_frustum.append(line)
            
    def _update_view_limits(self):
        """Update the view limits smoothly to keep the trajectory in view."""
        if len(self.trajectory_x) > 1:
            # Calculate current bounds
            min_x, max_x = min(self.trajectory_x), max(self.trajectory_x)
            min_y, max_y = min(self.trajectory_y), max(self.trajectory_y)
            min_z, max_z = min(self.trajectory_z), max(self.trajectory_z)
            
            # Add margin
            margin_x = max((max_x - min_x) * 0.2, self.margin)
            margin_y = max((max_y - min_y) * 0.2, self.margin)
            margin_z = max((max_z - min_z) * 0.2, self.margin)
            
            # Get current view limits
            curr_xlim = self.ax.get_xlim()
            curr_ylim = self.ax.get_ylim()
            curr_zlim = self.ax.get_zlim()
            
            # Calculate target limits
            target_xlim = [min_x - margin_x, max_x + margin_x]
            target_ylim = [min_y - margin_y, max_y + margin_y]
            target_zlim = [min_z - margin_z, max_z + margin_z]
            
            # Ensure minimum view range
            for target_lim in [target_xlim, target_ylim, target_zlim]:
                range_size = target_lim[1] - target_lim[0]
                if range_size < self.min_view_range:
                    center = (target_lim[1] + target_lim[0]) / 2
                    target_lim[0] = center - self.min_view_range / 2
                    target_lim[1] = center + self.min_view_range / 2
            
            # Smoothly update limits
            new_xlim = [curr_xlim[0] + self.smooth_factor * (target_xlim[0] - curr_xlim[0]),
                       curr_xlim[1] + self.smooth_factor * (target_xlim[1] - curr_xlim[1])]
            new_ylim = [curr_ylim[0] + self.smooth_factor * (target_ylim[0] - curr_ylim[0]),
                       curr_ylim[1] + self.smooth_factor * (target_ylim[1] - curr_ylim[1])]
            new_zlim = [curr_zlim[0] + self.smooth_factor * (target_zlim[0] - curr_zlim[0]),
                       curr_zlim[1] + self.smooth_factor * (target_zlim[1] - curr_zlim[1])]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.ax.set_zlim(new_zlim)

def draw_features_and_matches(frame: np.ndarray, keypoints: list, pose: np.ndarray = None) -> np.ndarray:
    """Draw detected features and pose information on the frame."""
    vis_frame = frame.copy()
    
    # Draw keypoints
    for kp in keypoints:
        pt = tuple(map(int, kp.pt))
        cv2.circle(vis_frame, pt, 3, (0, 255, 0), -1)
    
    # Draw pose information if available
    if pose is not None:
        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Draw coordinate axes
        length = 50
        origin = (50, 50)
        
        # Project coordinate axes
        points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        points_transformed = (R @ points.T + t.reshape(3, 1)).T
        
        # Draw axes
        for i, (color, point) in enumerate(zip([(0, 0, 255), (0, 255, 0), (255, 0, 0)], points_transformed[1:])):
            pt = tuple(map(int, [point[0] + origin[0], point[1] + origin[1]]))
            cv2.line(vis_frame, origin, pt, color, 2)
    
    return vis_frame

def get_image_paths(input_path: str) -> list:
    """Get sorted list of image paths from directory."""
    if os.path.isdir(input_path):
        # Look for common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(image_paths)
    return []

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ORB-SLAM3')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to camera configuration file')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input video file, camera index, or image directory')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save trajectory (optional)')
    parser.add_argument('--viz_delay', type=float, default=0.05,
                      help='Delay between frames for visualization (seconds)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    camera_matrix, dist_coeffs = create_camera_matrix(config)
    
    # Check if input is a directory (image sequence)
    image_paths = get_image_paths(args.input)
    if image_paths:
        print(f"Found {len(image_paths)} images in sequence")
        input_source = image_paths
    else:
        # Try video file or camera
        try:
            cap = cv2.VideoCapture(int(args.input))
        except ValueError:
            cap = cv2.VideoCapture(args.input)
            
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.input}")
            return
        input_source = cap
        
    # Initialize SLAM system
    slam = ORBSlam3(args.config)
    if not slam.initialize():
        print("Error: Failed to initialize SLAM system")
        return
    
    # Create windows for visualization
    cv2.namedWindow('ORB-SLAM3: Features', cv2.WINDOW_NORMAL)
    
    # Initialize 3D trajectory visualizer
    trajectory_vis = TrajectoryVisualizer()
        
    try:
        frame_idx = 0
        while True:
            # Read frame
            if isinstance(input_source, list):
                if frame_idx >= len(input_source):
                    break
                frame = cv2.imread(input_source[frame_idx])
                frame_idx += 1
            else:
                ret, frame = input_source.read()
                if not ret:
                    break
                
            # Process frame
            pose, keypoints = slam.process_frame(frame)
            
            # Visualize
            if pose is not None:
                # Draw features and pose
                vis_frame = draw_features_and_matches(frame, keypoints, pose)
                cv2.imshow('ORB-SLAM3: Features', vis_frame)
                
                # Update 3D trajectory visualization
                trajectory_vis.update(pose)
                
                # Add delay for visualization
                plt.pause(args.viz_delay)
                
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nSLAM system interrupted by user")
    finally:
        # Cleanup
        if not isinstance(input_source, list):
            input_source.release()
        cv2.destroyAllWindows()
        slam.shutdown()
        plt.close('all')
        
        # Save trajectory if requested
        if args.output and slam.get_trajectory():
            trajectory = slam.get_trajectory()
            np.save(args.output, trajectory)
            print(f"Trajectory saved to {args.output}")

if __name__ == "__main__":
    main() 