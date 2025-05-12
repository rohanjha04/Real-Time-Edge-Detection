#include <opencv2/opencv.hpp>
#include "EdgeDetector.hpp"

int main(int argc, char** argv) {
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

    cv::namedWindow("Original and Edge Detection", cv::WINDOW_NORMAL);
    EdgeDetector detector;

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000.0 / fps);
    if (fps <= 0.0 || fps > 120.0) delay = 30;  // Fallback

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        Image input = Image::fromMat(frame);
        Image edges = detector.process(input);
        cv::Mat edgeMat = edges.toMat();

        // Convert grayscale edge image to BGR for concatenation
        cv::Mat edgesBGR;
        cv::cvtColor(edgeMat, edgesBGR, cv::COLOR_GRAY2BGR);

        // Concatenate side by side: [original | edge]
        cv::Mat combined;
        cv::hconcat(frame, edgesBGR, combined);

        cv::imshow("Original and Edge Detection", combined);

        if (cv::waitKey(delay) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
