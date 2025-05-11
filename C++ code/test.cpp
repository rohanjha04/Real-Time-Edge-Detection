#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

// Global temporal buffers for evaluation
cv::Mat smoothedEdges;
cv::Mat prevGray;
bool firstCall = true;

double lowerRatio = 0.66;
double upperRatio = 1.33;
double alpha = 0.2; // temporal smoothing weight

// Wrap your preprocessing + Canny + temporal smoothing into one function
cv::Mat detectEdges(const cv::Mat& inputGray) {
    cv::Mat claheGray, blurred, edges;
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));

    // 1. Contrast
    clahe->apply(inputGray, claheGray);
    // 2. Blur
    cv::GaussianBlur(claheGray, blurred, cv::Size(5,5), 1.0);
    // 3. Adaptive Canny
    double m = cv::mean(blurred)[0];
    double lower = std::max(0.0, lowerRatio * m);
    double upper = std::min(255.0, upperRatio * m);
    cv::Canny(blurred, edges, lower, upper);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));

    // 4. Temporal smoothing
    cv::Mat edgesFloat;
    edges.convertTo(edgesFloat, CV_32F);
    if (firstCall) {
        edgesFloat.copyTo(smoothedEdges);
        inputGray.copyTo(prevGray);
        firstCall = false;
    } else {
        cv::accumulateWeighted(edgesFloat, smoothedEdges, alpha);
        inputGray.copyTo(prevGray);
    }
    cv::Mat display;
    smoothedEdges.convertTo(display, CV_8U);
    return display;
}

// Compute Precision, Recall, F1 given detection and vector of GT masks
void computePRF(const cv::Mat& det, const std::vector<cv::Mat>& gts,
                double& P, double& R, double& F1) {
    cv::Mat gtUnion = cv::Mat::zeros(det.size(), CV_8U);
    for (const auto &gt: gts) {
        gtUnion |= gt;
    }
    cv::Mat tpMat = det & gtUnion;
    cv::Mat fpMat = det & (~gtUnion);
    cv::Mat fnMat = (~det) & gtUnion;

    double TP = cv::countNonZero(tpMat);
    double FP = cv::countNonZero(fpMat);
    double FN = cv::countNonZero(fnMat);
    P = TP / (TP + FP + 1e-8);
    R = TP / (TP + FN + 1e-8);
    F1 = 2 * (P * R) / (P + R + 1e-8);
}

// Load all ground-truth edge maps for an image
std::vector<cv::Mat> loadGT(const fs::path &gtDir) {
    std::vector<cv::Mat> maps;
    for (auto &p: fs::directory_iterator(gtDir)) {
        if (p.path().extension() == ".png" || p.path().extension() == ".jpg") {
            cv::Mat m = cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE);
            if (!m.empty()) {
                // Binarize if needed
                cv::Mat bin;
                cv::threshold(m, bin, 128, 255, cv::THRESH_BINARY);
                maps.push_back(bin);
            }
        }
    }
    return maps;
}

int runEvaluation(const std::string &imgFolder, const std::string &gtFolder) {
    std::ofstream log("bsds_results.csv");
    log << "image,threshold,P,R,F1,ms\n";

    for (auto &imgPath : fs::directory_iterator(imgFolder)) {
        if (!imgPath.is_regular_file()) continue;
        cv::Mat img = cv::imread(imgPath.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        // Reset temporal buffers per image
        firstCall = true;

        // prepare GT folder for this image base name
        std::string base = imgPath.path().stem().string();
        fs::path thisGT = fs::path(gtFolder) / (base + "");
        auto gtMaps = loadGT(thisGT);
        if (gtMaps.empty()) continue;

        for (double t=10; t<=150; t+=5) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat edges = detectEdges(img);
            cv::Mat bin;
            cv::threshold(edges, bin, t, 255, cv::THRESH_BINARY);
            auto t1 = std::chrono::high_resolution_clock::now();

            double P,R,F1;
            computePRF(bin, gtMaps, P, R, F1);
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            log << base << "," << t << "," << P << "," << R << "," << F1 << "," << ms << "\n";
        }
    }

    log.close();
    std::cout << "Evaluation done. See bsds_results.csv\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--eval") {
        std::string split = argv[2];       // "test", "train", or "val"
        std::string baseDir = "archive";  // root of Kaggle download
        std::string imgFolder = baseDir + "/images/" + split;
        std::string gtFolder  = baseDir + "/ground_truth/" + split;
        return runEvaluation(imgFolder, gtFolder);
    }

    // Otherwise fallback to video loop
    std::string videoPath = "/Users/rohanjha/Desktop/AI-ML/Drishti/Tech/Videos/self_defense.mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) { std::cerr << "Cannot open video\n"; return -1; }

    cv::Mat frame;
    std::cout << "Press 'q' to quit\n";
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat edges = detectEdges(gray);
        cv::imshow("Edges", edges);
        if ((char)cv::waitKey(1) == 'q') break;
    }
    return 0;
}
