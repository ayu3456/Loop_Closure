import cv2
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import urllib.request
import zipfile
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

class LoopClosureDetector:
    def __init__(self, video_dir="input_videos", output_dir="output"):
        """Initialize the Loop Closure Detector"""
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.image_dir = output_dir  # Add back image_dir for visualization
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize SIFT detector with fewer features
        self.sift = cv2.SIFT_create(nfeatures=500)  # Reduced from 1000
        
        # Initialize FLANN matcher with fewer trees
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)  # Reduced from 5
        search_params = dict(checks=30)  # Reduced from 50
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Parameters for loop closure
        self.similarity_threshold = 0.4  # Increased threshold
        self.min_inlier_ratio = 0.15     # Increased slightly
        self.min_matches = 5             # Kept the same
        
        # Storage for frame data
        self.frames = []
        self.descriptors = []
        self.similarity_matrix = None
        self.frame_times = []  # Store timestamps for 3D trajectory
        self.frame_positions = []  # Store estimated positions for 3D trajectory

    def load_frames_from_directory(self, frames_dir):
        """Load frames from a directory containing frame_*.jpg files"""
        try:
            print(f"\nLoading frames from directory: {frames_dir}")
            
            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
            
            # Process every 10th frame
            step = 10
            frame_count = 0
            for i, frame_file in enumerate(frame_files):
                if i % step != 0:
                    continue
                
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    keypoints, descriptors = self.detect_and_compute(frame)
                    self.frames.append((frame, keypoints))
                    self.descriptors.append(descriptors)
                    # Calculate timestamp based on frame number (assuming 30 FPS)
                    frame_num = int(frame_file.split('_')[1].split('.')[0])
                    self.frame_times.append(frame_num / 30.0)
                    frame_count += 1
            
            print(f"\nTotal frames loaded: {frame_count}")
            print(f"(Processed every {step}th frame)")
            return frame_count
            
        except Exception as e:
            print(f"Error loading frames: {e}")
            return 0

    def create_3d_trajectory(self, loop_closures, output_path=None):
        """Create 3D trajectory visualization using Plotly"""
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Add trajectory points
        x = np.array(self.frame_times)  # Use time as X axis
        y = np.zeros(len(self.frames))  # Y axis (could be used for position)
        z = np.zeros(len(self.frames))  # Z axis (could be used for position)
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=2),
            name='Trajectory'
        ))
        
        # Add loop closure connections
        for i, j, _ in loop_closures:
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[j]],
                y=[y[i], y[j]],
                z=[z[i], z[j]],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Loop {i}-{j}'
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Trajectory with Loop Closures',
            scene=dict(
                xaxis_title='Time (seconds)',
                yaxis_title='Position Y',
                zaxis_title='Position Z'
            ),
            showlegend=True
        )
        
        # Save the plot
        if output_path is None:
            output_path = os.path.join(self.output_dir, '3d_trajectory.html')
        
        fig.write_html(output_path)
        print(f"\nSaved 3D trajectory visualization to: {output_path}")
        
        # Show the plot
        try:
            fig.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
            print("The visualization has been saved to disk. You can open it manually.")

    def create_distinctive_pattern(self, img, center, color):
        """Create a distinctive pattern at the given center"""
        x, y = center
        size = 40
        # Draw a cross
        cv2.line(img, (x-size, y), (x+size, y), color, 3)
        cv2.line(img, (x, y-size), (x, y+size), color, 3)
        # Draw a circle
        cv2.circle(img, (x, y), size//2, color, 2)
        # Draw a square
        cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, 2)

    def download_sample_images(self):
        """Generate synthetic test images that demonstrate loop closure"""
        # Create a base image with distinctive patterns
        base_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add multiple distinctive patterns
        patterns = [
            ((100, 100), (0, 255, 0)),   # Green pattern top-left
            ((540, 100), (255, 0, 0)),   # Red pattern top-right
            ((100, 380), (0, 0, 255)),   # Blue pattern bottom-left
            ((540, 380), (255, 255, 0)),  # Yellow pattern bottom-right
            ((320, 240), (255, 255, 255)) # White pattern center
        ]
        
        for center, color in patterns:
            self.create_distinctive_pattern(base_img, center, color)
        
        # Add text
        cv2.putText(base_img, "LOOP", (250, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(base_img, "CLOSURE", (230, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Generate a sequence of images with transformations
        num_frames = 10
        for i in range(num_frames):
            save_path = os.path.join(self.image_dir, f"frame_{i+1:03d}.jpg")
            
            if not os.path.exists(save_path):
                # For the last two frames, use transformations similar to first two frames
                if i >= num_frames - 2:
                    base_idx = i - (num_frames - 2)
                    angle = base_idx * 30 / num_frames + 2  # Slightly different angle
                    scale = 1.0 - (base_idx * 0.03) - 0.02  # Slightly different scale
                    tx = base_idx * 10 + 5  # Slightly different translation
                    ty = base_idx * 5 + 5
                else:
                    angle = i * 30 / num_frames
                    scale = 1.0 - (i * 0.03)
                    tx = i * 10
                    ty = i * 5
                
                rows, cols = base_img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation
                transformed = cv2.warpAffine(base_img, M, (cols, rows))
                
                # Add some random noise
                noise = np.random.normal(0, 2, transformed.shape).astype(np.uint8)
                transformed = cv2.add(transformed, noise)
                
                # Save the image
                cv2.imwrite(save_path, transformed)
                print(f"Generated: frame_{i+1:03d}.jpg")
                
        print("Generated test sequence with loop closure")

    def detect_and_compute(self, img):
        """Detect keypoints and compute descriptors"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Filter keypoints to reduce number
        if len(keypoints) > 100:  # Keep only top 100 strongest keypoints
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:100]
            descriptors = np.array([desc for kp, desc in zip(keypoints, descriptors)])
        
        return keypoints, descriptors
    
    def calculate_similarity(self, i, j):
        """Calculate similarity between two frames using feature matching"""
        if i == j:
            return 1.0  # Perfect similarity for same frame
        
        # Get descriptors
        desc1 = self.descriptors[i]
        desc2 = self.descriptors[j]
        
        if desc1 is None or desc2 is None:
            return 0.0
        
        # Match features
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Calculate similarity score
        num_matches = len(good_matches)
        if num_matches < self.min_matches:
            return 0.0
        
        # Calculate inlier ratio
        if len(matches) > 0:
            inlier_ratio = num_matches / len(matches)
        else:
            inlier_ratio = 0.0
        
        # Return similarity score
        return inlier_ratio

    def compute_similarity_score(self, matches, num_features1, num_features2):
        """Compute similarity score between two frames based on matches"""
        if not matches or num_features1 == 0 or num_features2 == 0:
            return 0.0
        
        # Use a more robust similarity metric
        # Consider both the number of matches and their quality
        match_ratio = len(matches) / min(num_features1, num_features2)
        quality_score = np.mean([m.distance for m in matches]) if matches else 1.0
        
        # Combine both factors
        similarity = match_ratio * (1.0 / (1.0 + quality_score))
        
        return similarity

    def visualize_loop_closure(self, frame_idx1, frame_idx2, matches, mask):
        """Visualize loop closure between two frames"""
        frame1_path = os.path.join(self.image_dir, f"frame_{frame_idx1+1:03d}.jpg")
        frame2_path = os.path.join(self.image_dir, f"frame_{frame_idx2+1:03d}.jpg")
                            
                            # Get inlier ratio
                            inlier_ratio = np.sum(mask) / len(matches)
                            
                            if inlier_ratio > self.min_inlier_ratio:
                                # Visualize loop closure
                                try:
                                    self.visualize_loop_closure(i, j, matches, mask)
                                except Exception as e:
                                    print(f"Error in visualization: {e}")
                                
                                # Add to loop closures
                                loop_closures.append((i, j, self.similarity_matrix[i, j]))
                                print(f"Found loop closure between frames {i+1} and {j+1}")
                                
                                if len(loop_closures) >= 5:  # Limit to 5 loop closures for visualization
                continue
            
            # Calculate similarity
            similarity = self.calculate_similarity(i, j)
            self.similarity_matrix[i, j] = similarity
            self.similarity_matrix[j, i] = similarity
    
    def detect_loop_closures(self):
        """Detect loop closures between frames"""
        if len(self.frames) < 2:
            return []
        
        print("\nComputing similarity matrix...")
        
        # Create similarity matrix
        n = len(self.frames)
        self.similarity_matrix = np.zeros((n, n))
        
        # Compare each pair of frames
        for i in range(n):
            for j in range(i + 1, n):
                if i == j:
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(i, j)
                self.similarity_matrix[i, j] = similarity
                self.similarity_matrix[j, i] = similarity
        
        # Find loop closures
        loop_closures = []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.similarity_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    loop_closures.append((i, j, similarity))
        
        return loop_closures

    # Find loop closures
    loop_closures = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = self.similarity_matrix[i, j]
            if similarity >= self.similarity_threshold:
                loop_closures.append((i, j, similarity))
    
    # Perform geometric verification
    verified_loop_closures = []
    num_frames = detector.load_frames_from_directory(frames_dir)
    if num_frames == 0:
        print(f"Warning: Could not load any frames from {frames_dir}")
        return
    
    # Detect loop closures
    loop_closures = detector.detect_loop_closures()
    
    # Create 3D trajectory visualization with a unique filename
    output_path = os.path.join(detector.output_dir, f'3d_trajectory_{video_file}.html')
    
    # Create 3D trajectory visualization
    detector.create_3d_trajectory(loop_closures, output_path)
    
    print("\nLoop Closure Detection Summary:")
    print(f"Total loop closures found: {len(loop_closures)}")
    for i, j, similarity in loop_closures:
        print(f"Loop closure between frames {i+1} and {j+1} (similarity: {similarity:.3f})")
    
    # Clear frames for next video
    detector.frames = []
    detector.descriptors = []
    detector.similarity_matrix = None
    detector.frame_times = []
    detector.frame_positions = []
    
    print(f"\nFound {len(video_files)} videos to process:")
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        video_path = os.path.join(detector.video_dir, video_file)
        
        # Get full path to video
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_file)
        
        # Extract frames from video
        num_frames = detector.extract_frames_from_video(video_path)
        if num_frames == 0:
            print(f"Warning: Could not extract any frames from {video_path}")
            continue
        
        # Detect loop closures
        loop_closures = detector.detect_loop_closures()
        
        # Create 3D trajectory visualization
        detector.create_3d_trajectory(loop_closures)
        
        print("\nLoop Closure Detection Summary:")
        print(f"Total loop closures found: {len(loop_closures)}")
        for i, j, similarity in loop_closures:
            print(f"Loop closure between frames {i+1} and {j+1} (similarity: {similarity:.3f})")
        
        # Clear frames for next video
        detector.frames = []
        detector.descriptors = []
        detector.similarity_matrix = None
        detector.frame_times = []
        detector.frame_positions = []

if __name__ == "__main__":
    main() 