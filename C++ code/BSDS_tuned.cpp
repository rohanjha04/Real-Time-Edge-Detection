#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

// Globals
static bool firstIteration = true;
static cv::Mat smoothedEdges;
static bool trainingMode = false;

// Temporal smoothing factor
const double defaultAlpha = 0.25;

// Ratio candidates (lowThresh = r.first * mean; highThresh = r.second * mean)
static const std::vector<std::pair<double,double>> ratioCandidates = {
    {0.5,1.5}, {0.55,1.6}, {0.6,1.7}, {0.6,1.8}, {0.65,1.85}, {0.7,1.9}
};

// CLAHE instance
static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));


//─── PREPROCESS (CLAHE + blur) ──────────────────────────────────────────────
cv::Mat preprocess(const cv::Mat& gray) {
    cv::Mat claheGray, blurred;
    clahe->apply(gray, claheGray);
    cv::GaussianBlur(claheGray, blurred, cv::Size(5,5), 1.5);
    return blurred;  // <<< FIX: return the processed image
}


//─── EDGE DETECTION (Canny + temporal smoothing) ────────────────────────────
cv::Mat detectEdges(const cv::Mat& blurred, double lowThresh, double highThresh) {
    cv::Mat edges;
    cv::Canny(blurred, edges, lowThresh, highThresh);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));

    if (!trainingMode) {
        // temporal smoothing
        cv::Mat edgesF;
        edges.convertTo(edgesF, CV_32F);
        if (firstIteration) {
            edgesF.copyTo(smoothedEdges);
            firstIteration = false;
        } else {
            cv::accumulateWeighted(edgesF, smoothedEdges, defaultAlpha);
        }
        cv::Mat display;
        smoothedEdges.convertTo(display, CV_8U);
        return display;
    }
    return edges;
}


//─── LOAD GT EDGES AS 1-PX POINTS ────────────────────────────────────────────
std::vector<cv::Point> loadGTEdges(const fs::path& gtFolder, const std::string& name) {
    std::vector<cv::Point> pts;
    for (auto& entry : fs::directory_iterator(gtFolder)) {
        std::string fn = entry.path().filename().string();
        if (!entry.is_regular_file() ||
            fn.rfind(name + "_", 0) != 0 ||
            entry.path().extension() != ".jpg")
            continue;

        cv::Mat m = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (m.empty()) continue;
        cv::Mat bin_pre;
        cv::threshold(m, bin_pre, 128, 255, cv::THRESH_BINARY);
        cv::Mat bin;
        cv::dilate(bin_pre, bin, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));
        for (int y = 0; y < bin.rows; ++y)
            for (int x = 0; x < bin.cols; ++x)
                if (bin.at<uchar>(y,x))
                    pts.emplace_back(x,y);
    }
    std::cout << "[loadGT] Loaded " << pts.size() << " GT edge pixels for " << name << "\n";
    return pts;
}


//─── COMPUTE P, R, F1 WITH RADIUS-BASED MATCHING ─────────────────────────────
void computePRF(const cv::Mat& det, 
                const std::vector<cv::Point>& gtPts,
                double& P, double& R, double& F1)
{
    const int RADIUS = 5;
    int H = det.rows, W = det.cols;

    // Build a GT map for fast lookup & removal
    cv::Mat gtMap = cv::Mat::zeros(H, W, CV_8U);
    for (auto& p : gtPts)
        gtMap.at<uchar>(p.y, p.x) = 1;

    int TP = 0, FP = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (!det.at<uchar>(y,x)) continue;
            bool matched = false;
            for (int dy = -RADIUS; dy <= RADIUS && !matched; ++dy) {
                for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy<0||yy>=H||xx<0||xx>=W) continue;
                    if (gtMap.at<uchar>(yy,xx)) {
                        TP++;
                        gtMap.at<uchar>(yy,xx) = 0;  // consume
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) FP++;
        }
    }
    int FN = cv::countNonZero(gtMap);
    P  = TP / double(TP + FP + 1e-8);
    R  = TP / double(TP + FN + 1e-8);
    F1 = 2 * (P * R) / (P + R + 1e-8);
}


//─── PROCESS A VIDEO FRAME ──────────────────────────────────────────────────
cv::Mat processFrame(const cv::Mat& frame, cv::Mat& prevGray) {
    // grayscale + motion mask
    cv::Mat gray, motionMask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    if (!trainingMode && !prevGray.empty()) {
        cv::absdiff(gray, prevGray, motionMask);
        cv::threshold(motionMask, motionMask, 30, 255, cv::THRESH_BINARY);
    }
    gray.copyTo(prevGray);

    // edges
    cv::Mat blurred = preprocess(gray);
    double m = cv::mean(blurred)[0];
    double lowT  = std::max(0.0, ratioCandidates[1].first  * m);  // placeholder
    double highT = std::min(255.0, ratioCandidates[1].second * m);
    cv::Mat edges = detectEdges(blurred, lowT, highT);

    if (!motionMask.empty())
        cv::bitwise_or(edges, motionMask, edges);

    cv::Mat colorE, out;
    cv::cvtColor(edges, colorE, cv::COLOR_GRAY2BGR);
    cv::hconcat(frame, colorE, out);
    return out;
}


//─── EVALUATION LOOP ────────────────────────────────────────────────────────
int runEvaluation(const std::string& split, bool reportBest) {
    trainingMode = true;
    std::string base = "archive";
    std::string imgF = base + "/images/" + split;
    std::string gtF  = base + "/converted_ground_truth/" + split;
    std::ofstream log("bsds_" + split + ".csv");
    log << "image,lowRatio,highRatio,P,R,F1,ms\n";

    std::map<std::pair<double,double>, double> sumF;
    for (auto&r:ratioCandidates) sumF[r]=0;
    int n=0;

    for (auto& p: fs::directory_iterator(imgF)) {
        std::string name = p.path().stem().string();
        cv::Mat img = cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        n++; firstIteration=true;

        auto gtPts = loadGTEdges(gtF, name);
        if (gtPts.empty()) continue;

        cv::Mat blurred = preprocess(img);
        double m = cv::mean(blurred)[0];

        for (auto& r : ratioCandidates) {
            double lowT  = std::max(0.0, r.first  * m);
            double highT = std::min(255.0, r.second * m);

            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat edges = detectEdges(blurred, lowT, highT);
            auto t1 = std::chrono::high_resolution_clock::now();

            double P,R,F1;
            computePRF(edges, gtPts, P,R,F1);
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

            sumF[r] += F1;
            log << name << "," << r.first << "," << r.second << ","
                << P << "," << R << "," << F1 << "," << ms << "\n";
        }
    }
    log.close();

    if (reportBest && n) {
        double bestF=0; std::pair<double,double> bestR;
        for (auto&kv:sumF) {
            double avgF = kv.second / n;
            if (avgF > bestF) { bestF=avgF; bestR=kv.first; }
        }
        std::cout << "Best ratios("<<split<<")=("
                  <<bestR.first<<","<<bestR.second
                  <<") avgF1="<<bestF<<"\n";
    }
    return 0;
}


//─── MAIN ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc==2 && std::string(argv[1])=="--train")
        return runEvaluation("train", true);
    if (argc==3 && std::string(argv[1])=="--eval")
        return runEvaluation(argv[2], false);

    // Video mode
    trainingMode = false;
    cv::VideoCapture cap("/Users/rohanjha/Desktop/AI-ML/Drishti/Tech/Videos/self_defense.mp4");
    if (!cap.isOpened()) { std::cerr<<"Cannot open video\n"; return -1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps<=0) fps=30;
    double delayMs = 1000.0 / fps;

    cv::Mat frame, prevGray;
    std::cout<<"Press 'q' to quit.\n";
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::Mat out = processFrame(frame, prevGray);
        cv::imshow("Original + Edges", out);
        if ((char)cv::waitKey(int(delayMs))=='q') break;
    }
    return 0;
}
