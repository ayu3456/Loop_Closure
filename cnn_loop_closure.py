import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import cv2
import os
from random_forest_loop_closure import RandomForestLoopClosureDetector

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Use ResNet18 as base model, pre-trained on ImageNet
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze the parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.features(x)

class CNNLoopClosureDetector(RandomForestLoopClosureDetector):
    def __init__(self, image_dir="images/loop"):
        super().__init__(image_dir)
        
        # Initialize CNN feature extractor
        self.cnn = CNNFeatureExtractor()
        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda()
        self.cnn.eval()

        # Override model and scaler filenames for CNN-specific versions
        self.model_filename = "cnn_rf_loop_closure_model.joblib"
        self.scaler_filename = "cnn_rf_scaler.joblib"

        # Reload model and scaler if specific CNN versions exist, otherwise it will use RF ones or train anew
        if os.path.exists(self.model_filename) and os.path.exists(self.scaler_filename):
            print(f"Loading pre-trained CNN-RF model from {self.model_filename}")
            self.rf_classifier = joblib.load(self.model_filename)
            print(f"Loading CNN-RF scaler from {self.scaler_filename}")
            self.scaler = joblib.load(self.scaler_filename)
        else:
            print(f"No pre-trained CNN-specific model/scaler found at {self.model_filename} / {self.scaler_filename}. Ensure they are generated or training will occur.")
        
        # Image preprocessing for CNN
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_cnn_features(self, image):
        """Extract CNN features from an image"""
        # Convert OpenCV BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        # Apply transformations
        input_tensor = self.transform(pil_image)
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
            
        with torch.no_grad():
            features = self.cnn(input_batch)
            
        # Convert to numpy array and flatten
        features = features.cpu().numpy().flatten()
        return features
        
    def process_frames(self):
        """Process all frames and store their features"""
        # Clear previous frame data for this new processing batch
        self.frames = []
        self.sift_keypoints = []
        self.sift_descriptors = []
        self.cnn_features_list = [] # Assuming cnn_features are stored per frame in a list
        self.frame_times = []

        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"No image files (.jpg, .jpeg, .png) found in {self.image_dir}")
            return
            
        print(f"\nProcessing {len(image_files)} frames from {self.image_dir}...")
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(self.image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue
                
            # Extract both SIFT and CNN features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            cnn_features = self.extract_cnn_features(image)
            
            self.frames.append({
                'id': image_file, # Store filename as ID
                'image': image, # Consider not storing full image if memory is an issue
                'keypoints': keypoints,
                'descriptors': descriptors,
                'cnn_features': cnn_features
            })
            self.sift_keypoints.append(keypoints)
            self.sift_descriptors.append(descriptors)
            self.cnn_features_list.append(cnn_features) # Store CNN features in the list

            # Populate frame_times, assuming filename format like 'prefix_NUMBER.ext'
            try:
                # Extracts the last number sequence before the extension
                frame_num_str = ''.join(filter(str.isdigit, image_file.split('_')[-1].split('.')[0]))
                if frame_num_str:
                    frame_num = int(frame_num_str)
                    self.frame_times.append(frame_num / 30.0)  # Assume 30 FPS for timestamp
                else:
                    # Fallback if no number found, use index
                    print(f"Warning: Could not parse frame number from {image_file}. Using index {i-1} for frame_time.")
                    self.frame_times.append((i-1) / 30.0) # i is 1-based index from enumerate
            except Exception as e:
                print(f"Warning: Error parsing frame number from {image_file}: {e}. Using index {i-1} for frame_time.")
                self.frame_times.append((i-1) / 30.0)

            if i % 50 == 0 or i == len(image_files):
                 print(f"Processed frame {i}/{len(image_files)}: {image_file}")
            
    def compute_cnn_similarity(self, features1, features2):
        """Compute cosine similarity between CNN features"""
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
    def extract_features_for_pair(self, matches, num_features1, num_features2, distances, cnn_sim):
        """Extract combined SIFT and CNN features"""
        if not matches or len(matches) < 2:
            return np.zeros(9)  # One additional feature for CNN similarity
            
        match_distances = [m.distance for m in matches]
        match_ratio = len(matches) / ((num_features1 + num_features2) / 2)
        distance_stats = np.percentile(match_distances, [25, 50, 75]) if match_distances else [float('inf')] * 3
        spatial_stats = np.percentile(distances, [25, 50, 75]) if len(distances) > 0 else [float('inf')] * 3
        
        features = [
            len(matches),  # Number of matches
            match_ratio,   # Match ratio
            np.mean(match_distances),  # Mean match distance
            np.std(match_distances),   # Standard deviation of match distances
            np.min(distances) if len(distances) > 0 else float('inf'),  # Minimum spatial distance
            distance_stats[1],  # Median match distance
            spatial_stats[1],   # Median spatial distance
            match_ratio * np.exp(-np.mean(match_distances)/100),  # Combined score
            cnn_sim  # CNN similarity score
        ]
        
        return np.array(features)
        
    def detect_loop_closures_with_rf(self):
        """Detect loop closures using combined CNN and random forest"""
        if not self.frames:
            print("No frames processed. Running frame processing...")
            self.process_frames()
            
        n = len(self.frames)
        if n < 2:
            print("Not enough frames to detect loop closures")
            return []
            
        features_list = []
        pairs_list = []
        
        print("\nExtracting features for loop closure detection...")
        for i in range(n):
            for j in range(i + 1, n):
                if j - i <= 2:  # Skip consecutive frames
                    continue
                    
                # Compute CNN similarity
                cnn_sim = self.compute_cnn_similarity(
                    self.frames[i]['cnn_features'],
                    self.frames[j]['cnn_features']
                )
                
                desc1 = self.frames[i]['descriptors']
                desc2 = self.frames[j]['descriptors']
                
                if desc1 is None or desc2 is None:
                    continue
                    
                # SIFT Feature Matching (knnMatch + Ratio Test)
                matches = [] # Initialize to empty list
                # Descriptors desc1 and desc2 are already checked for None earlier in this method.
                # We also need to ensure they are not empty and are of type float32 for FLANN.
                if desc1.shape[0] > 0 and desc2.shape[0] > 0:
                    # Ensure descriptors are float32 (SIFT descriptors usually are, but good practice)
                    d1_processed = desc1 if desc1.dtype == np.float32 else np.float32(desc1)
                    d2_processed = desc2 if desc2.dtype == np.float32 else np.float32(desc2)

                    raw_matches_list = self.matcher.knnMatch(d1_processed, d2_processed, k=2)
                    
                    for m_tuple in raw_matches_list:
                        # knnMatch returns a list of k matches for each descriptor.
                        # We need at least two matches (m and n) for the ratio test.
                        if len(m_tuple) == 2:
                            k_match1, k_match2 = m_tuple
                            if k_match1.distance < 0.7 * k_match2.distance: # Lowe's Ratio test
                                matches.append(k_match1)
                # If descriptors are empty or no good matches found, 'matches' will remain an empty list.
                print(f"\nFrame pair ({i+1}, {j+1}):")
                print(f"Number of matches: {len(matches)}")
                print(f"CNN similarity: {cnn_sim:.3f}")
                
                if matches:
                    pts1 = np.float32([self.frames[i]['keypoints'][m.queryIdx].pt for m in matches])
                    pts2 = np.float32([self.frames[j]['keypoints'][m.trainIdx].pt for m in matches])
                    distances = np.linalg.norm(pts1 - pts2, axis=1)
                    print(f"Average spatial distance: {np.mean(distances):.2f}")
                else:
                    distances = []
                    print("No matches found")
                
                features = self.extract_features_for_pair(
                    matches,
                    len(self.frames[i]['keypoints']),
                    len(self.frames[j]['keypoints']),
                    distances,
                    cnn_sim
                )
                
                features_list.append(features)
                pairs_list.append((i, j))
                
        if not features_list:
            print("No valid feature pairs found")
            return []
            
        # Scale features
        X = self.scaler.transform(np.array(features_list))
        
        # Predict loop closures
        predictions = self.rf_classifier.predict_proba(X)[:, 1]
        
        # Find loop closures
        loop_closures = []
        for idx, (i, j) in enumerate(pairs_list):
            print(f"\nChecking pair ({i+1}, {j+1}):")
            print(f"Confidence score: {predictions[idx]:.3f}")
            
            # Combined confidence using both RF and CNN similarity
            cnn_sim = self.compute_cnn_similarity(
                self.frames[i]['cnn_features'],
                self.frames[j]['cnn_features']
            )
            combined_conf = 0.7 * predictions[idx] + 0.3 * cnn_sim
            print(f"Combined confidence: {combined_conf:.3f}")
            
            if combined_conf > 0.3:  # Threshold for combined confidence
                print("Passed confidence threshold, checking geometric consistency...")
                desc1 = self.frames[i]['descriptors']
                desc2 = self.frames[j]['descriptors']
                
                # SIFT Feature Matching (knnMatch + Ratio Test) for geometric verification
                matches = [] # Initialize to empty list
                if desc1 is not None and desc2 is not None and desc1.shape[0] > 0 and desc2.shape[0] > 0:
                    # Ensure descriptors are float32
                    d1_processed = desc1 if desc1.dtype == np.float32 else np.float32(desc1)
                    d2_processed = desc2 if desc2.dtype == np.float32 else np.float32(desc2)

                    raw_matches_list = self.matcher.knnMatch(d1_processed, d2_processed, k=2)
                    
                    for m_tuple in raw_matches_list:
                        if len(m_tuple) == 2:
                            k_match1, k_match2 = m_tuple # Using k_match1, k_match2 to avoid 'n' conflict
                            if k_match1.distance < 0.7 * k_match2.distance: # Lowe's Ratio test
                                matches.append(k_match1)
                # If descriptors are None, empty or no good matches found, 'matches' will remain an empty list.
                
                if len(matches) >= self.min_matches:
                    pts1 = np.float32([self.frames[i]['keypoints'][m.queryIdx].pt for m in matches])
                    pts2 = np.float32([self.frames[j]['keypoints'][m.trainIdx].pt for m in matches])
                    
                    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                    
                    if H is not None:
                        inlier_ratio = np.sum(mask) / len(mask)
                        print(f"Inlier ratio: {inlier_ratio:.3f}")
                        if inlier_ratio > 0.3:
                            loop_closures.append((i, j, combined_conf, inlier_ratio))
                            self.visualize_loop_closure(i, j, matches, mask)
                            print("Loop closure detected!")
                    else:
                        print("Failed to find homography")
                else:
                    print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            else:
                print("Failed confidence threshold")
                
        return loop_closures

def main():
    # Initialize detector
    detector = CNNLoopClosureDetector()
    
    # Generate or load sample images
    detector.download_sample_images()
    
    # Process frames
    detector.process_frames()
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    n_samples = 2000
    
    # Generate feature distributions
    n_matches = np.random.normal(75, 15, n_samples)
    match_ratios = np.random.normal(0.7, 0.15, n_samples)
    match_distances = np.random.normal(100, 20, n_samples)
    distance_stds = np.random.normal(15, 5, n_samples)
    spatial_distances = np.random.normal(95, 20, n_samples)
    cnn_similarities = np.random.normal(0.8, 0.15, n_samples)  # CNN similarities
    
    # Combine features
    synthetic_features = np.column_stack([
        np.abs(n_matches),
        np.clip(match_ratios, 0, 1),
        np.abs(match_distances),
        np.abs(distance_stds),
        np.abs(spatial_distances),
        np.abs(match_distances) * 0.8,
        np.abs(spatial_distances) * 0.9,
        np.clip(match_ratios, 0, 1) * np.exp(-np.abs(match_distances)/100),
        np.clip(cnn_similarities, 0, 1)  # CNN similarity feature
    ])
    
    # Generate labels with combined scoring
    quality_score = (
        0.3 * np.clip(n_matches / 100, 0, 1) +
        0.2 * np.clip(match_ratios, 0, 1) +
        0.2 * np.exp(-np.abs(match_distances)/150) +
        0.3 * np.clip(cnn_similarities, 0, 1)  # CNN contribution
    )
    synthetic_labels = (quality_score > 0.6).astype(int)
    
    # Train random forest
    print("\nTraining random forest classifier...")
    detector.train_random_forest(synthetic_features, synthetic_labels)
    
    # Detect loop closures
    print("\nDetecting loop closures using CNN + Random Forest...")
    loop_closures = detector.detect_loop_closures_with_rf()
    
    # Print results
    print("\nDetected loop closures:")
    if loop_closures:
        for i, j, confidence, inlier_ratio in loop_closures:
            print(f"Frames {i+1} and {j+1}: Combined Confidence = {confidence:.3f}, Inlier ratio = {inlier_ratio:.3f}")
    else:
        print("No loop closures detected")

if __name__ == "__main__":
    main() 