import cv2
import os
import datetime
import sys

def capture_video(output_dir="input_videos", frame_rate=30, duration=30):
    """
    Capture video from camera and save frames
    
    Args:
        output_dir (str): Directory to save frames
        frame_rate (int): Frames per second
        duration (int): Duration of recording in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video capture with default camera (0)
    print("\nAttempting to open default camera (index 0)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        print("Please check:")
        print("1. The camera is properly connected")
        print("2. You have granted camera permissions to Python")
        print("3. No other application is using the camera")
        print("\nTrying to open camera with different settings...")
        
        # Try different backend settings
        backend = cv2.CAP_AVFOUNDATION  # macOS specific backend
        cap = cv2.VideoCapture(0, backend)
        
        if not cap.isOpened():
            print("\nFailed to open camera with alternative settings")
            sys.exit(1)
        else:
            print("\nSuccessfully opened camera with alternative settings")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera properties:")
    print(f"Resolution: {width}x{height}")
    print(f"Frame rate: {frame_rate} FPS")
    print(f"Duration: {duration} seconds")
    print("\nPress 'q' to stop early")
    
    # Try to set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    
    # Verify settings
    print(f"\nActual camera settings:")
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Create a window with a larger size
    cv2.namedWindow('Video Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Capture', 800, 600)
    
    print(f"Starting video capture. Press 'q' to stop early.")
    print(f"Resolution: {width}x{height}")
    
    # Capture frames
    frame_count = 0
    start_time = datetime.datetime.now()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Display frame
        cv2.imshow('Video Capture', frame)
        
        # Exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Stop after specified duration
        if elapsed_time >= duration:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nRecording complete! Captured {frame_count} frames")
    print(f"Frames saved in: {output_dir}")

if __name__ == "__main__":
    # Create a timestamped directory for this recording
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"input_videos/{timestamp}_recording"
    
    # Capture video for 30 seconds
    capture_video(output_dir=output_dir, duration=30)
    
    # After recording, run loop closure detection
    print("\nRunning loop closure detection...")
    try:
        from loop_closure_detector import LoopClosureDetector
        detector = LoopClosureDetector(image_dir=output_dir)
        detector.detect_loop_closure()
    except Exception as e:
        print(f"\nError running loop closure detection: {str(e)}")
        print("Frames have been saved successfully. You can run the detector manually later.")
