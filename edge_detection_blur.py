import cv2
import numpy as np

def auto_canny_thresholds(image, sigma=0.33):
    # Compute the median of the pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection thresholds using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper

def main():
    # Open the default camera (0 usually corresponds to the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set a smaller resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Check for CUDA support in your build of OpenCV
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    canny_gpu = None
    if use_gpu:
        try:
            canny_gpu = cv2.cuda.createCannyEdgeDetector(100, 200)
            print("GPU acceleration is available. Using CUDA.")
        except cv2.error as e:
            print("CUDA not properly supported in this build:", e)
            use_gpu = False
    else:
        print("GPU acceleration is not available. Using CPU instead.")
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (quality enhancement)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Option: Adaptive threshold using the auto Canny technique
        lower, upper = auto_canny_thresholds(blurred, sigma=0.33)
        # For experiments, you may choose to use:
        # edges = cv2.Canny(blurred, lower, upper)
        # Otherwise, using fixed thresholds:
        # edges = cv2.Canny(blurred, 100, 200)

        if use_gpu:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(blurred)
            gpu_edges = canny_gpu.detect(gpu_mat)
            edges = gpu_edges.download()
        else:
            edges = cv2.Canny(blurred, 100, 200)
        
        # Optional: Refine the edge image using dilation (improves edge continuity)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Display the resulting frame
        cv2.imshow('Real-Time Edge Detection', edges)
        
        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
