#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera
    cv::VideoCapture cap("/Users/rohanjha/Desktop/AI-ML/Drishti/Tech/Videos/self_defense.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam!" << std::endl;
        return -1;
    }

    // Set resolution (optional)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame, gray, claheGray, blurred, edges;
    cv::Mat smoothedEdges; // Accumulated edge map for temporal smoothing
    cv::Mat prevGray;      // Previous gray frame for motion detection
    cv::Mat motionMask;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    bool firstIteration = true;
    const double alpha = 0.2; // Weight for accumulateWeighted (lower = more smoothing)

    std::cout << "Press 'q' to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        // Preprocessing: convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // For motion detection, compute the absolute difference with previous frame
        if (!firstIteration) {
            cv::absdiff(gray, prevGray, motionMask);
            cv::threshold(motionMask, motionMask, 30, 255, cv::THRESH_BINARY);
        }
        prevGray = gray.clone();

        // Apply CLAHE to improve contrast before edge detection
        clahe->apply(gray, claheGray);

        // Apply Gaussian blur to reduce noise
        cv::GaussianBlur(claheGray, blurred, cv::Size(5, 5), 1.0);

        // Compute Canny edges using adaptive thresholds (here based on a fraction of the mean intensity)
        double median = cv::mean(blurred)[0];
        double lower = std::max(0.0, 0.66 * median);
        double upper = std::min(255.0, 1.33 * median);
        cv::Canny(blurred, edges, lower, upper);

        // Optional dilation to enhance continuity in edges
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(edges, edges, kernel);

        // Initialize the smoothedEdges on the first iteration
        if (firstIteration) {
            edges.convertTo(smoothedEdges, CV_32F);
            firstIteration = false;
        } else {
            // Accumulate weighted: blend current edge map with previous smoothed result
            cv::Mat edgesFloat;
            edges.convertTo(edgesFloat, CV_32F);
            cv::accumulateWeighted(edgesFloat, smoothedEdges, alpha);
        }
        // Convert back to 8-bit image
        cv::Mat displayEdges;
        smoothedEdges.convertTo(displayEdges, CV_8U);

        // For moving object capture, combine the edges with the motion mask.
        // This makes moving edges brighter.
        if (!motionMask.empty()) {
            cv::Mat combined;
            cv::bitwise_or(displayEdges, motionMask, combined);
            displayEdges = combined;
        }

        // Show the result
        cv::imshow("Temporal Smoothed Edge Detection", displayEdges);

        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
