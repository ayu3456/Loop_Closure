import cv2
import os
import datetime

def record_video(output_dir="recorded_video", duration=30):
    """Record video for specified duration"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nCamera properties:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Recording duration: {duration} seconds")
    
    # Create window
    cv2.namedWindow('Video Recording', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Recording', 800, 600)
    
    # Calculate total frames
    total_frames = int(fps * duration)
    print(f"\nStarting recording. Press 'q' to stop early")
    print(f"Total frames to capture: {total_frames}")
    
    frame_count = 0
    start_time = datetime.datetime.now()
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
            
        cv2.imshow('Video Recording', frame)
        
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        progress = (frame_count / total_frames) * 100
        print(f"\rRecording... {progress:.1f}% complete ({frame_count}/{total_frames} frames)", end="")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nRecording stopped early")
            break
            
        frame_count += 1
    
    print(f"\n\nRecording complete! Captured {frame_count} frames")
    print(f"Frames saved to: {output_dir}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video(duration=30)
    
    print("\nRunning loop closure detection...")
    try:
        from loop_closure_detector import LoopClosureDetector
        detector = LoopClosureDetector(image_dir="recorded_video")
        loop_closures = detector.process_sequence()
        print("\nLoop Closure Detection Summary:")
        print(f"Total loop closures found: {len(loop_closures)}")
        for i, j, similarity in loop_closures:
            print(f"Loop closure between frames {i+1} and {j+1} (similarity: {similarity:.3f})")
    except Exception as e:
        print(f"\nError running loop closure detection: {str(e)}")
        print("You can run the detector manually later using the recorded frames in 'recorded_video' directory")
        print("To run manually, use:")
        print("python3 loop_closure_detector.py")
