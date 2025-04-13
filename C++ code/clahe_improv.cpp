#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, gray, clahe_gray, blurred, edges, previousEdges;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    std::cout << "Press 'q' to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocessing
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        clahe->apply(gray, clahe_gray);
        cv::GaussianBlur(clahe_gray, blurred, cv::Size(5, 5), 1.0);

        // Auto Canny using median
        double median = cv::mean(blurred)[0];
        double lower = std::max(0.0, 0.66 * median);
        double upper = std::min(255.0, 1.33 * median);
        cv::Canny(blurred, edges, lower, upper);

        // Optional: Smooth transition between frames (reduce flicker)
        if (!previousEdges.empty()) {
            cv::addWeighted(edges, 0.7, previousEdges, 0.3, 0, edges);
        }
        previousEdges = edges.clone();

        // Thin dilation to preserve structure
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(edges, edges, kernel);

        cv::imshow("Stable Canny Edge Detection", edges);

        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
