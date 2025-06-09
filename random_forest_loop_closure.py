import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2
import os
import joblib
from loop_closure_detector import LoopClosureDetector

class RandomForestLoopClosureDetector(LoopClosureDetector):
    def __init__(self, image_dir="images/loop"):
        super().__init__(image_dir)
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.frames = []
        self.model_filename = "rf_loop_closure_model.joblib"
        self.scaler_filename = "rf_scaler.joblib"

        # Load pre-trained model and scaler if they exist
        if os.path.exists(self.model_filename) and os.path.exists(self.scaler_filename):
            print(f"Loading pre-trained model from {self.model_filename}")
            self.rf_classifier = joblib.load(self.model_filename)
            print(f"Loading scaler from {self.scaler_filename}")
            self.scaler = joblib.load(self.scaler_filename)
        else:
            print("No pre-trained model/scaler found. Will train from scratch if needed.")
        
    def process_frames(self):
        """Process all frames and store their keypoints and descriptors"""
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        print("\nProcessing frames...")
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(self.image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue
                
            # Detect keypoints and compute descriptors
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            self.frames.append({
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors
            })
            print(f"Processed frame {i}/{len(image_files)}")
        
    def extract_features_for_pair(self, matches, num_features1, num_features2, distances):
        """Extract features for random forest classification"""
        if not matches or len(matches) < 2:
            return np.zeros(8)
            
        match_distances = [m.distance for m in matches]
        
        # Calculate additional statistics
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
            match_ratio * np.exp(-np.mean(match_distances))  # Combined score
        ]
        
        return np.array(features)
    
    def train_random_forest(self, training_data, labels):
        """Train the random forest classifier"""
        # Fit the scaler with the training data and transform it
        X = self.scaler.fit_transform(training_data)
        print(f"Saving scaler to {self.scaler_filename}")
        joblib.dump(self.scaler, self.scaler_filename)

        # Train the Random Forest classifier
        self.rf_classifier.fit(X, labels)
        print(f"Saving trained RF model to {self.model_filename}")
        joblib.dump(self.rf_classifier, self.model_filename)
    
    def detect_loop_closures_with_rf(self):
        """Detect loop closures using random forest classification"""
        if not self.frames:
            print("No frames processed. Running frame processing...")
            self.process_frames()
            
        n = len(self.frames)
        if n < 2:
            print("Not enough frames to detect loop closures")
            return []
            
        features_list = []
        pairs_list = []
        
        # Extract features for all frame pairs
        print("\nExtracting features for random forest classification...")
        for i in range(n):
            for j in range(i + 1, n):
                if j - i <= 2:  # Skip consecutive frames
                    continue
                    
                desc1 = self.frames[i]['descriptors']
                desc2 = self.frames[j]['descriptors']
                
                if desc1 is None or desc2 is None:
                    continue
                    
                matches = self.match_features(desc1, desc2)
                print(f"\nFrame pair ({i+1}, {j+1}):")
                print(f"Number of matches: {len(matches)}")
                
                if matches:
                    # Calculate spatial distances between matched keypoints
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
                    distances
                )
                
                features_list.append(features)
                pairs_list.append((i, j))
        
        if not features_list:
            print("No valid feature pairs found")
            return []
            
        # Scale features
        X = self.scaler.transform(np.array(features_list))
        
        # Predict loop closures
        predictions = self.rf_classifier.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Find loop closures based on random forest predictions
        loop_closures = []
        for idx, (i, j) in enumerate(pairs_list):
            print(f"\nChecking pair ({i+1}, {j+1}):")
            print(f"Confidence score: {predictions[idx]:.3f}")
            
            if predictions[idx] > 0.05:  # Lower confidence threshold since we're using stricter training data
                print("Passed confidence threshold, checking geometric consistency...")
                # Verify with geometric consistency
                desc1 = self.frames[i]['descriptors']
                desc2 = self.frames[j]['descriptors']
                matches = self.match_features(desc1, desc2)
                
                if len(matches) >= self.min_matches:
                    pts1 = np.float32([self.frames[i]['keypoints'][m.queryIdx].pt for m in matches])
                    pts2 = np.float32([self.frames[j]['keypoints'][m.trainIdx].pt for m in matches])
                    
                    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                    
                    if H is not None:
                        inlier_ratio = np.sum(mask) / len(mask)
                        print(f"Inlier ratio: {inlier_ratio:.3f}")
                        if inlier_ratio > 0.3:
                            loop_closures.append((i, j, predictions[idx], inlier_ratio))
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
    rf_detector = RandomForestLoopClosureDetector()
    
    # Generate or load sample images
    rf_detector.download_sample_images()
    
    # Process frames
    rf_detector.process_frames()
    
    # Generate better synthetic training data
    print("\nGenerating synthetic training data...")
    n_samples = 2000
    
    # Generate more realistic feature distributions based on observed data
    n_matches = np.random.normal(75, 15, n_samples)  # Adjusted based on observed data
    match_ratios = np.random.normal(0.7, 0.15, n_samples)
    match_distances = np.random.normal(100, 20, n_samples)
    distance_stds = np.random.normal(15, 5, n_samples)
    spatial_distances = np.random.normal(95, 20, n_samples)
    
    # Combine features
    synthetic_features = np.column_stack([
        np.abs(n_matches),
        np.clip(match_ratios, 0, 1),
        np.abs(match_distances),
        np.abs(distance_stds),
        np.abs(spatial_distances),
        np.abs(match_distances) * 0.8,
        np.abs(spatial_distances) * 0.9,
        np.clip(match_ratios, 0, 1) * np.exp(-np.abs(match_distances)/100)
    ])
    
    # Generate labels with more lenient distribution
    quality_score = (
        0.4 * np.clip(n_matches / 100, 0, 1) +  # More matches is better
        0.3 * np.clip(match_ratios, 0, 1) +     # Higher match ratio is better
        0.3 * np.exp(-np.abs(match_distances)/150)  # Lower match distances are better
    )
    synthetic_labels = (quality_score > 0.6).astype(int)
    
    # Train random forest
    print("\nTraining random forest classifier...")
    rf_detector.train_random_forest(synthetic_features, synthetic_labels)
    
    # Detect loop closures using random forest
    print("\nDetecting loop closures using random forest...")
    loop_closures = rf_detector.detect_loop_closures_with_rf()
    
    # Print results
    print("\nDetected loop closures:")
    if loop_closures:
        for i, j, confidence, inlier_ratio in loop_closures:
            print(f"Frames {i+1} and {j+1}: Confidence = {confidence:.3f}, Inlier ratio = {inlier_ratio:.3f}")
    else:
        print("No loop closures detected")

if __name__ == "__main__":
    main() 