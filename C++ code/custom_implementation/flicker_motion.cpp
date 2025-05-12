#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// Global for edge smoothing
static bool firstIteration = true;
static cv::Mat smoothedEdges;

// Parameters
const double alpha = 0.25;

// Detect edges with CLAHE, blur, adaptive Canny, dilation, temporal smoothing
cv::Mat detectEdges(const cv::Mat& gray) {
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat claheGray, blurred, edges;

    clahe->apply(gray, claheGray);
    cv::GaussianBlur(claheGray, blurred, cv::Size(5, 5), 1.0);

    double m = cv::mean(blurred)[0];
    double lower = std::max(0.0, 0.66 * m);
    double upper = std::min(255.0, 1.33 * m);
    cv::Canny(blurred, edges, lower, upper);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

    cv::Mat edgesF;
    edges.convertTo(edgesF, CV_32F);
    if (firstIteration) {
        edgesF.copyTo(smoothedEdges);
        firstIteration = false;
    } else {
        cv::accumulateWeighted(edgesF, smoothedEdges, alpha);
    }
    cv::Mat display;
    smoothedEdges.convertTo(display, CV_8U);
    return display;
}

// Process one frame: compute motion, detect edges, combine frames
cv::Mat processFrame(const cv::Mat& frame, cv::Mat& prevGray) {
    cv::Mat gray, motionMask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Motion mask
    if (!firstIteration) {
        cv::absdiff(gray, prevGray, motionMask);
        cv::threshold(motionMask, motionMask, 30, 255, cv::THRESH_BINARY);
    }
    gray.copyTo(prevGray);

    // Edge detection
    cv::Mat edges = detectEdges(gray);

    // Combine motion mask
    if (!motionMask.empty()) {
        cv::bitwise_or(edges, motionMask, edges);
    }

    // Convert edges to color
    cv::Mat edgesColor;
    cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2BGR);  // Restore this line

    // Resize frames to avoid issues on portrait videos
    int displayWidth = 640; // Width to scale to
    int displayHeight = 480; // Height to scale to

    // Resize frames to avoid window overflow
    cv::Mat resizedFrame = frame.clone(); 
    cv::Mat resizedEdgesColor = edgesColor.clone();

    if (!frame.empty()) {
        cv::resize(resizedFrame, resizedFrame, cv::Size(displayWidth, displayHeight));
    }

    if (!edgesColor.empty()) {
        cv::resize(edgesColor, resizedEdgesColor, cv::Size(displayWidth, displayHeight));
    }

    // Ensure no empty frames before concatenating
    cv::Mat combined;
    if (!resizedFrame.empty() && !resizedEdgesColor.empty()) {
        cv::hconcat(resizedFrame, resizedEdgesColor, combined);
    }

    return combined;
}

int main() {
    cv::VideoCapture cap("C:/Users/Sujatha/Videos/OnePlus/VID20220807105126.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video!" << std::endl;
        return -1;
    }

    // Optional resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Frame timing
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    double frameDurationMs = 1000.0 / fps;

    cv::Mat frame, prevGray;

    std::cout << "Press 'q' to quit." << std::endl;
    while (true) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty()) break;

        // Check if the frame is portrait and rotate it if necessary
        if (frame.cols < frame.rows) {
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        }

        // Process the frame
        cv::Mat displayFrame = processFrame(frame, prevGray);
        cv::imshow("Original and Edge Detection", displayFrame);

        auto t1 = std::chrono::high_resolution_clock::now();
        double procMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int delay = std::max(1, static_cast<int>(frameDurationMs - procMs));
        if ((char)cv::waitKey(delay) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
