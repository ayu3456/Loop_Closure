import cv2
import numpy as np
import plotly.graph_objects as go
import os

class LoopClosureDetector:
    def __init__(self, video_dir="input_videos", output_dir="output"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.image_dir = output_dir
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize SIFT detector with fewer features
        self.sift = cv2.SIFT_create(nfeatures=500)
        
        # Initialize FLANN matcher with fewer trees
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=30)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Parameters for loop closure
        self.similarity_threshold = 0.4
        self.min_inlier_ratio = 0.15
        self.min_matches = 5
        
        # Storage for frame data
        self.frames = []
        self.descriptors = []
        self.similarity_matrix = None
        self.frame_times = []
        self.frame_positions = []

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
        if len(keypoints) > 100:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:100]
            descriptors = np.array([desc for kp, desc in zip(keypoints, descriptors)])
        
        return keypoints, descriptors

    def calculate_similarity(self, i, j):
        """Calculate similarity between two frames using feature matching"""
        if i == j:
            return 1.0
        
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
        
        return inlier_ratio

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

    def create_3d_trajectory(self, loop_closures, output_path=None):
        """Create 3D trajectory visualization using Plotly"""
        # Calculate position estimates using both consecutive frames and loop closures
        positions = []
        
        # Initialize first position at origin
        positions.append((0, 0, 0))
        
        # Store previous position for each frame
        prev_positions = {0: (0, 0, 0)}
        
        # Process all frames
        for i in range(1, len(self.frames)):
            try:
                # Get keypoints and descriptors
                keypoints1 = self.frames[i-1][1]
                keypoints2 = self.frames[i][1]
                
                # Get points from keypoints
                points1 = np.float32([kp.pt for kp in keypoints1])
                points2 = np.float32([kp.pt for kp in keypoints2])
                
                # Find homography
                H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Calculate translation from homography
                    # Extract translation components
                    tx = H[0, 2]
                    ty = H[1, 2]
                    tz = 0  # Assuming movement is primarily in 2D plane
                    
                    # Add to previous position
                    prev_x, prev_y, prev_z = prev_positions[i-1]
                    
                    # Scale the translation
                    scale = 1.0  # Fixed scale for consecutive frames
                    position = (prev_x + tx * scale, 
                               prev_y + ty * scale, 
                               prev_z + tz * scale)
                    
                    # Update position for this frame
                    prev_positions[i] = position
                    positions.append(position)
                
                # Check if this frame is part of any loop closure
                for i1, i2, similarity in loop_closures:
                    if i1 == i:
                        # Get keypoints and descriptors for loop closure
                        keypoints1 = self.frames[i1][1]
                        keypoints2 = self.frames[i2][1]
                        
                        # Get points from keypoints
                        points1 = np.float32([kp.pt for kp in keypoints1])
                        points2 = np.float32([kp.pt for kp in keypoints2])
                        
                        # Find homography
                        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            # Calculate translation from homography
                            tx = H[0, 2]
                            ty = H[1, 2]
                            tz = 0
                            
                            # Scale the translation based on similarity
                            scale = similarity * 100
                            
                            # Calculate loop closure position
                            loop_position = (prev_x + tx * scale, 
                                           prev_y + ty * scale, 
                                           prev_z + tz * scale)
                            
                            # Add loop closure position
                            positions.append(loop_position)
                            
            except Exception as e:
                print(f"Error calculating position for frame {i}: {e}")
                positions.append(positions[-1])
        
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Add trajectory points
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        z = [pos[2] for pos in positions]
        
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
                xaxis_title='Position X',
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

def main():
    """Main function to run the loop closure detection on video frames"""
    detector = LoopClosureDetector()
    
    # Process specific video
    video_file = "20250520_180039_recording"
    frames_dir = os.path.join(detector.video_dir, video_file)
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        return
    
    print(f"\nProcessing frames from directory: {video_file}")
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    
    # Process every 10th frame
    step = 10
    frame_count = 0
    
    # Create windows for display
    cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    
    # Initialize trajectory visualization
    trajectory_img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    for i, frame_file in enumerate(frame_files):
        if i % step != 0:
            continue
            
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        # Process frame
        keypoints, descriptors = detector.detect_and_compute(frame)
        detector.frames.append((frame, keypoints))
        detector.descriptors.append(descriptors)
        
        # Calculate timestamp based on frame number (assuming 30 FPS)
        frame_num = int(frame_file.split('_')[1].split('.')[0])
        detector.frame_times.append(frame_num / 30.0)
        frame_count += 1
        
        # Draw keypoints on frame
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
        
        # Display processed frame
        cv2.imshow('Processed Frame', frame_with_keypoints)
        
        # Update trajectory visualization
        if len(detector.frame_times) > 1:
            # Get current position (using frame number as x-coordinate)
            x = int(frame_num / 30.0 * 100)  # Convert time to pixels
            y = 400  # Fixed y-coordinate for simplicity
            
            # Draw trajectory line
            if len(detector.frame_times) > 2:
                cv2.line(trajectory_img, 
                         (int(detector.frame_times[-2]*100), 400),
                         (x, y),
                         (0, 255, 0), 2)
            
            # Draw current position
            cv2.circle(trajectory_img, (x, y), 5, (0, 0, 255), -1)
            
        # Display trajectory
        cv2.imshow('Trajectory', trajectory_img)
        
        # Wait for 1ms and check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\nTotal frames processed: {frame_count}")
    print(f"(Processed every {step}th frame)")
    
    # Close windows
    cv2.destroyAllWindows()
    
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

if __name__ == "__main__":
    main()
