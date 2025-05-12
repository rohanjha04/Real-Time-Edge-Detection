#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam!" << std::endl;
        return -1;
    }

    // Set resolution (optional)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, gray, blurred, edges;
    std::cout << "Press 'q' to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Apply Gaussian blur
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
        // Canny edge detection
        cv::Canny(blurred, edges, 100, 200);
        // Optional: Dilation
        cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

        // Show the result
        cv::imshow("Edge Detection", edges);

        // Exit on 'q' key
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
