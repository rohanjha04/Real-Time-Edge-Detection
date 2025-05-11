import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")

    # Initialize previous edge frame for temporal smoothing
    prev_edges = None
    alpha = 0.6  # Weight for current frame in temporal smoothing

    # CLAHE setup
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        equalized = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 1.5)

        # Canny edge detection with tuned thresholds
        low_thresh = 50
        high_thresh = 150
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Morphological closing to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Temporal smoothing with previous frame
        if prev_edges is not None:
            edges = cv2.addWeighted(edges, alpha, prev_edges, 1 - alpha, 0)
        prev_edges = edges.copy()

        # Show output
        cv2.imshow('Stable Real-Time Edge Detection', edges)

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
