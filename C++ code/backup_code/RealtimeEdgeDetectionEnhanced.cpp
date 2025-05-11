#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Open input: webcam by default, or video file if provided
    cv::VideoCapture cap;
    if (argc > 1) {
        cap.open(argv[1]);  // Open video file
    } else {
        cap.open(0);        // Open default webcam
    }
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open input!" << std::endl;
        return -1;
    }

    // Create a resizable window
    cv::namedWindow("Original and Edge Detection", cv::WINDOW_NORMAL);

    cv::Mat frame, gray, blurred, edges, edges_bgr, combined, display;

    // Scaling factor to reduce size (75% of original)
    double display_scale = 0.75;

    while (true) {
        // Capture a frame
        cap >> frame;
        if (frame.empty()) {
            break;  // End of video or webcam disconnected
        }

        // Process the frame for edge detection
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.8);
        cv::Canny(blurred, edges, 30, 90);
        cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);

        // Combine original and edge-detected frames side by side
        cv::hconcat(frame, edges_bgr, combined);

        // Resize the combined image to 75% of its original size
        int new_width = static_cast<int>(combined.cols * display_scale);
        int new_height = static_cast<int>(combined.rows * display_scale);
        cv::resize(combined, display, cv::Size(new_width, new_height));

        // Display the resized image
        cv::imshow("Original and Edge Detection", display);

        // Exit on 'q' key press
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}