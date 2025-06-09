import numpy as np
import cv2
import threading
import os
from cnn_loop_closure import CNNLoopClosureDetector # For CNN-based loop closure
from typing import Optional, Dict, List, Tuple
from tracking import Tracker

class ORBSlam3:
    """
    Main class for ORB-SLAM3 system that coordinates all the SLAM components.
    """
    def __init__(self, config_path: str):
        """
        Initialize ORB-SLAM3 system.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.is_running = False
        
        # Load configuration
        self.load_config()
        
        # Initialize components
        self.feature_extractor = None
        self.tracker = None
        self.mapper = None
        
        # Initialize system state
        self.current_frame = None
        self.keyframes: List = []
        self.map_points: Dict = {}
        self.trajectory: List[np.ndarray] = []
        self.current_pose = np.eye(4)
        
        # Keyframe parameters
        self.min_keyframe_rotation = 0.15  # radians
        self.min_keyframe_translation = 0.1  # meters
        self.last_keyframe_pose = np.eye(4)
        self.keyframe_counter = 0

        # Loop Closure components and parameters
        self.keyframe_image_dir = "orb_slam_keyframes"
        os.makedirs(self.keyframe_image_dir, exist_ok=True)
        # Ensure model/scaler paths are correct if cnn_loop_closure.py is not in CWD when this runs
        self.cnn_loop_detector = CNNLoopClosureDetector(image_dir=self.keyframe_image_dir)
        self.loop_closure_trigger_interval = 10 # Run loop closure check every N keyframes
        self.keyframes_since_last_lc = 0
        
        # Threading locks
        self.map_update_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        
    def load_config(self):
        """Load camera configuration."""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        camera_config = config['Camera']
        self.camera_matrix = np.array([
            [camera_config['fx'], 0, camera_config['cx']],
            [0, camera_config['fy'], camera_config['cy']],
            [0, 0, 1]
        ])
        
        self.dist_coeffs = np.array([
            camera_config['k1'],
            camera_config['k2'],
            camera_config['p1'],
            camera_config['p2'],
            camera_config['k3']
        ])

    def initialize(self) -> bool:
        """
        Initialize all SLAM system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize feature extraction
            self.feature_extractor = cv2.ORB_create(
                nfeatures=2000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=19,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
            
            # Initialize tracker
            self.tracker = Tracker(self.camera_matrix, self.dist_coeffs)
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize ORB-SLAM3: {str(e)}")
            return False

    def process_frame(self, frame: np.ndarray, depth: np.ndarray = None) -> Tuple[Optional[np.ndarray], List]:
        """
        Process a new frame through the SLAM system.
        
        Args:
            frame: Input RGB image frame
            depth: Optional depth image (in meters)
            
        Returns:
            Tuple[Optional[np.ndarray], List]: (Estimated camera pose matrix (4x4), Keypoints) or (None, []) if tracking failed
        """
        if not self.is_running:
            return None, []
            
        with self.pose_lock:
            self.current_frame = frame.copy()
            
            # 1. Extract features
            keypoints, descriptors = self.feature_extractor.detectAndCompute(frame, None)
            
            if self.tracker is not None:
                # 2. Track camera motion
                estimated_pose = self.tracker.track(frame, keypoints, descriptors, depth)
                
                if estimated_pose is not None:
                    # Update current pose
                    self.current_pose = estimated_pose
                    self.trajectory.append(self.current_pose)
                    
                    # Check if we should create a new keyframe
                    if self._need_new_keyframe(estimated_pose):
                        self._create_keyframe(frame, keypoints, descriptors, estimated_pose, depth)
                    
                    return self.current_pose, keypoints
            
            # If tracking failed or no tracker
            if len(self.trajectory) == 0:
                # First frame, initialize pose
                self.trajectory.append(self.current_pose)
                self._create_keyframe(frame, keypoints, descriptors, self.current_pose, depth)
            
            return self.current_pose, keypoints

    def _need_new_keyframe(self, current_pose: np.ndarray) -> bool:
        """
        Check if we need to create a new keyframe.
        
        Args:
            current_pose: Current camera pose
            
        Returns:
            bool: True if new keyframe needed
        """
        # Extract rotation and translation from relative transformation
        relative_transform = np.linalg.inv(self.last_keyframe_pose) @ current_pose
        R = relative_transform[:3, :3]
        t = relative_transform[:3, 3]
        
        # Compute rotation angle
        angle = np.arccos((np.trace(R) - 1) / 2)
        
        # Compute translation distance
        distance = np.linalg.norm(t)
        
        return angle > self.min_keyframe_rotation or distance > self.min_keyframe_translation
        
    def _create_keyframe(self, frame: np.ndarray, keypoints: List, 
                        descriptors: np.ndarray, pose: np.ndarray,
                        depth: np.ndarray = None):
        """
        Create a new keyframe and add it to the system.
        
        Args:
            frame: Image frame
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            pose: Camera pose
            depth: Optional depth image (in meters)
        """
        # Triangulate 3D points
        points_3d = np.zeros((len(keypoints), 3))
        
        if depth is not None:
            # Use depth information when available
            for i, kp in enumerate(keypoints):
                x, y = map(int, kp.pt)
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                    z = depth[y, x]
                    if z > 0:  # Valid depth
                        # Back-project to 3D using depth
                        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                        points_3d[i] = [x_3d, y_3d, z]
        else:
            # Fallback to simple projection when no depth available
            for i, kp in enumerate(keypoints):
                pt = np.array([kp.pt[0], kp.pt[1], 1.0])
                points_3d[i] = (np.linalg.inv(pose)[:3, :3] @ pt) + pose[:3, 3]
        
        # Update state
        self.last_keyframe_pose = pose.copy()
        self.keyframe_counter += 1
        self.keyframes.append({
            'id': self.keyframe_counter,
            'frame': frame.copy(), # Storing frame for now, consider path if memory is an issue
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'points_3d': points_3d
        })

        # Save keyframe image for CNN loop detector
        image_path = os.path.join(self.keyframe_image_dir, f"keyframe_{self.keyframe_counter:06d}.png")
        cv2.imwrite(image_path, frame)

        self.keyframes_since_last_lc += 1

        # Periodically run loop closure detection
        if self.keyframes_since_last_lc >= self.loop_closure_trigger_interval and self.keyframe_counter > self.loop_closure_trigger_interval:
            print(f"\n--- [ORB-SLAM3 System] Triggering Loop Closure Detection (Keyframe {self.keyframe_counter}) ---")
            # This tells the detector to re-scan its image_dir and update internal features
            self.cnn_loop_detector.process_frames()
            
            if len(self.cnn_loop_detector.frames) >= 2: # Need at least 2 frames for detection
                detected_loops = self.cnn_loop_detector.detect_loop_closures_with_rf()
                if detected_loops:
                    print(f"--- [ORB-SLAM3 System] CNN Loop Detector Found Potential Loops: ---")
                    for loop_info in detected_loops:
                        # loop_info is (idx1, idx2, combined_conf, inlier_ratio)
                        # These indices are for cnn_loop_detector's internal list. 
                        # Mapping them back to ORBSlam3's keyframe IDs requires care if cnn_loop_detector doesn't process all frames or processes them out of order.
                        # For now, assume cnn_loop_detector.frames[idx1]['id'] gives the filename like 'keyframe_00000X.png'
                        kf_id1_str = self.cnn_loop_detector.frames[loop_info[0]].get('id', f'UnknownID_idx{loop_info[0]}')
                        kf_id2_str = self.cnn_loop_detector.frames[loop_info[1]].get('id', f'UnknownID_idx{loop_info[1]}')
                        print(f"    Loop between {kf_id1_str} and {kf_id2_str} | Confidence: {loop_info[2]:.3f}, Inlier Ratio: {loop_info[3]:.3f}")
                    # TODO: Add actual loop correction mechanism (e.g., pose graph optimization)
                else:
                    print("--- [ORB-SLAM3 System] CNN Loop Detector: No loops found this cycle. ---")
            else:
                print("--- [ORB-SLAM3 System] CNN Loop Detector: Not enough processed frames to detect loops. ---")
            self.keyframes_since_last_lc = 0
            print("--- [ORB-SLAM3 System] Loop Closure Detection Cycle Complete ---\n")

    def shutdown(self):
        """
        Properly shutdown the SLAM system.
        """
        self.is_running = False
        
    def get_trajectory(self) -> List[np.ndarray]:
        """
        Get the camera trajectory.
        
        Returns:
            List[np.ndarray]: List of 4x4 pose matrices
        """
        return self.trajectory
        
    def get_map_points(self) -> Dict:
        """
        Get the current map points.
        
        Returns:
            Dict: Dictionary containing map points and their descriptors
        """
        with self.map_update_lock:
            return self.map_points.copy() 