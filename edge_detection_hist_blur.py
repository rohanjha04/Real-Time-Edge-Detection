import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    # Set a standard resolution for balance between performance and quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        #equalized = cv2.equalizeHist(gray)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        # cv2.imshow('Equalized and Blurred', blurred)

        # Apply Canny edge detection with manually chosen thresholds
        low_thresh = 100
        high_thresh = 200
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Optional: apply morphological dilation to strengthen edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Display the edges
        cv2.imshow('Enhanced Real-Time Edge Detection', edges)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
