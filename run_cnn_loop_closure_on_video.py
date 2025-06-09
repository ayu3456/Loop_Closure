#run_cnn_loop_closure_on_video.py
import argparse
import os
import cv2 # For frame extraction count
from cnn_loop_closure import CNNLoopClosureDetector # Assuming it's in the same directory or PYTHONPATH

def main(args):
    output_dir_base = args.output_base
    run_output_dir = None
    frames_source_dir = None
    extract_from_video = False

    if args.video_file:
        video_path = args.video_file
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        run_output_dir = os.path.join(output_dir_base, video_name + "_video_results")
        frames_source_dir = os.path.join(run_output_dir, "extracted_frames")
        os.makedirs(frames_source_dir, exist_ok=True)
        extract_from_video = True
        print(f"Processing video: {video_path}")
    elif args.frames_dir:
        frames_input_dir = args.frames_dir
        if not os.path.isdir(frames_input_dir):
            print(f"Error: Frames directory not found at {frames_input_dir}")
            return
        dir_name = os.path.basename(os.path.normpath(frames_input_dir))
        run_output_dir = os.path.join(output_dir_base, dir_name + "_frames_results")
        frames_source_dir = frames_input_dir # Use the provided directory directly
        os.makedirs(run_output_dir, exist_ok=True) # For trajectory and other outputs
        extract_from_video = False
        print(f"Processing frames from directory: {frames_input_dir}")
    else:
        print("Error: Either --video_file or --frames_dir must be specified.")
        return
    
    # Ensure run_output_dir is created if not already (e.g. if frames_source_dir is outside output_base)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"Initializing CNN Loop Closure Detector...")
    # The detector will use its default model paths to load the trained RF model and scaler
    # The detector will use frames_source_dir as its image_dir for processing
    # Its output_dir (from base class) will also be frames_source_dir
    detector = CNNLoopClosureDetector(image_dir=frames_source_dir)

    # Check if models are loaded (basic check)
    if detector.rf_classifier is None or not hasattr(detector.rf_classifier, 'predict_proba') or detector.scaler is None:
        print("Error: RF Classifier or Scaler not loaded properly in CNNLoopClosureDetector.")
        print(f"Please ensure '{detector.model_filename}' and '{detector.scaler_filename}' exist and are valid.")
        print("You might need to run the training script (e.g., main function in cnn_loop_closure.py) first.")
        return

    if extract_from_video:
        print(f"Extracting frames from {args.video_file} to {frames_source_dir}...")
        # Use the base class method for frame extraction
        # extract_frames_from_video is part of LoopClosureDetector, which CNNLoopClosureDetector inherits
        num_extracted_frames = detector.extract_frames_from_video(args.video_file, output_frames_dir=frames_source_dir)

        if num_extracted_frames == 0:
            print(f"No frames extracted from {args.video_file}. Check video path and format. Exiting.")
            return
        print(f"Successfully extracted {num_extracted_frames} frames.")
    else:
        # If using a pre-existing frames directory, count the frames (optional, for info)
        try:
            num_existing_frames = len([name for name in os.listdir(frames_source_dir) if os.path.isfile(os.path.join(frames_source_dir, name))])
            print(f"Using {num_existing_frames} frames from directory {frames_source_dir}.")
            if num_existing_frames == 0:
                print(f"No image files found in {frames_source_dir}. Exiting.")
                return
        except OSError as e:
            print(f"Error accessing frames directory {frames_source_dir}: {e}. Exiting.")
            return

    # Process these extracted frames (extracts SIFT and CNN features)
    print("Processing extracted frames (SIFT + CNN features)...")
    detector.process_frames() # This is the overridden method in CNNLoopClosureDetector

    if not detector.frames:
        print("No frames were processed by detector.process_frames(). Exiting.")
        return

    # Detect loop closures using the combined CNN+RF approach
    print("Detecting loop closures...")
    # This is the overridden method in CNNLoopClosureDetector
    # It uses the RF model trained on SIFT+CNN features
    loop_closures = detector.detect_loop_closures_with_rf() 

    print("\nDetected Loop Closures:")
    if loop_closures:
        for i_idx, j_idx, confidence, inlier_ratio in loop_closures:
            frame_id1 = detector.frames[i_idx].get('id', f"FrameIndex_{i_idx+1}")
            frame_id2 = detector.frames[j_idx].get('id', f"FrameIndex_{j_idx+1}")
            print(f"Loop between {frame_id1} and {frame_id2}: Confidence={confidence:.3f}, Inlier Ratio={inlier_ratio:.3f}")
    else:
        print("No loop closures detected.")

    # Populate frame_times based on the number of processed frames (assuming a fixed FPS for visualization)
    # This is needed by the base class's create_3d_trajectory method.
    detector.frame_times = [idx / 30.0 for idx in range(len(detector.frames))] # Assume 30 FPS for visualization time axis

    # Reformat loop_closures for the trajectory plot which expects (idx1, idx2, score_or_ignored_value)
    if loop_closures:
        loop_closures_for_plot = [(lc[0], lc[1], lc[2]) for lc in loop_closures] # Use first 3 elements
    else:
        loop_closures_for_plot = []

    trajectory_output_path = os.path.join(run_output_dir, "cnn_loop_closure_3d_trajectory.html")
    print(f"Creating 3D trajectory visualization at {trajectory_output_path}...")
    try:
        detector.create_3d_trajectory(loop_closures_for_plot, output_path=trajectory_output_path)
        print(f"Trajectory visualization saved. You can open it in a web browser.")
    except Exception as e:
        print(f"Error creating 3D trajectory: {e}")

    print(f"\nProcessing complete. Results and trajectory saved in {run_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN Loop Closure Detection on a video file or a directory of frames.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_file", type=str, help="Path to the input video file.")
    group.add_argument("--frames_dir", type=str, help="Path to the directory containing pre-extracted frames.")
    parser.add_argument("--output_base", type=str, default="output/video_run_results", help="Base directory for output.")
    
    args = parser.parse_args()
    main(args)
