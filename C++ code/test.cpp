#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

// Globals for video mode smoothing
static bool firstIteration = true;
static cv::Mat smoothedEdges;
const double temporalAlpha = 0.25;

// Auto-Canny parameter
const double sigma = 0.33;

//─── 1) PREPROCESS (CLAHE + GAUSSIAN BLUR) ───────────────────────────────────
cv::Mat preprocess(const cv::Mat& gray) {
    cv::Mat claheGray, blurred;
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
    clahe->apply(gray, claheGray);
    cv::GaussianBlur(claheGray, blurred, cv::Size(5,5), 1.5);
    return blurred;
}

//─── 2) AUTO-CANNY EDGE DETECTION ────────────────────────────────────────────
cv::Mat detectEdges(const cv::Mat& blurred) {
    // 2.1 compute median of blurred image
    std::vector<uchar> pix;
    pix.assign(blurred.datastart, blurred.dataend);
    auto mid = pix.begin() + pix.size()/2;
    std::nth_element(pix.begin(), mid, pix.end());
    double med = *mid;

    // 2.2 derive thresholds
    double low  = std::max(0.0,     (1.0 - sigma) * med);
    double high = std::min(255.0, (1.0 + sigma) * med);

    // 2.3 run Canny + dilate
    cv::Mat edges;
    cv::Canny(blurred, edges, low, high);
    cv::dilate(edges, edges, cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2,2)));

    // 2.4 temporal smoothing for video
    if (!smoothedEdges.empty()) {
        cv::Mat eF;
        edges.convertTo(eF, CV_32F);
        if (firstIteration) {
            eF.copyTo(smoothedEdges);
            firstIteration = false;
        } else {
            cv::accumulateWeighted(eF, smoothedEdges, temporalAlpha);
        }
        cv::Mat disp;
        smoothedEdges.convertTo(disp, CV_8U);
        return disp;
    }
    return edges;
}

//─── 3) LOAD & SKELETONIZE GT INTO A BINARY CV::Mat ─────────────────────────
cv::Mat loadGTSkel(const fs::path& gtFolder, const std::string& name) {
    cv::Mat accum;
    bool gotAny = false;
    for (auto& entry : fs::directory_iterator(gtFolder)) {
        std::string fn = entry.path().filename().string();
        if (!entry.is_regular_file() ||
            fn.rfind(name + "_", 0) != 0 ||
            entry.path().extension() != ".jpg")
            continue;

        cv::Mat m = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (m.empty()) continue;

        cv::Mat bin, skel;
        cv::threshold(m, bin, 128, 255, cv::THRESH_BINARY);
        cv::dilate(bin, skel, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));


        if (!gotAny) {
            accum = skel.clone();
            gotAny = true;
        } else {
            cv::bitwise_or(accum, skel, accum);
        }
    }
    return gotAny ? accum : cv::Mat();
}

//─── 4) DISTANCE-TRANSFORM PRF ────────────────────────────────────────────────
void computePRF_dt(const cv::Mat& det,
                   const cv::Mat& gtSkel,
                   double& P, double& R, double& F1)
{
    const int RADIUS = 5;
    CV_Assert(det.type()==CV_8U && gtSkel.type()==CV_8U);
    CV_Assert(det.size()==gtSkel.size());

    // build DT from GT edges
    cv::Mat invGT, dtGT;
    cv::bitwise_not(gtSkel, invGT);
    cv::distanceTransform(invGT, dtGT, cv::DIST_L2, 3);

    // count TP & FP
    int TP=0, FP=0;
    for (int y=0; y<det.rows; ++y) {
        const uchar* dR = det.ptr<uchar>(y);
        const float* dtR = dtGT.ptr<float>(y);
        for (int x=0; x<det.cols; ++x) {
            if (!dR[x]) continue;
            if (dtR[x] <= RADIUS) TP++;
            else                  FP++;
        }
    }

    // build DT from DET to count FN
    cv::Mat invDet, dtDet;
    cv::bitwise_not(det, invDet);
    cv::distanceTransform(invDet, dtDet, cv::DIST_L2, 3);

    int FN=0;
    for (int y=0; y<gtSkel.rows; ++y) {
        const uchar* gR  = gtSkel.ptr<uchar>(y);
        const float* d2R = dtDet.ptr<float>(y);
        for (int x=0; x<gtSkel.cols; ++x) {
            if (!gR[x]) continue;
            if (d2R[x] > RADIUS) FN++;
        }
    }

    P  = TP / double(TP + FP + 1e-8);
    R  = TP / double(TP + FN + 1e-8);
    F1 = 2 * (P * R) / (P + R + 1e-8);
}

//─── 5) VIDEO FRAME PROCESSING ───────────────────────────────────────────────
cv::Mat processFrame(const cv::Mat& frame, cv::Mat& prevGray) {
    cv::Mat gray, motion;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    if (!prevGray.empty()) {
        cv::absdiff(gray, prevGray, motion);
        cv::threshold(motion, motion, 30, 255, cv::THRESH_BINARY);
    }
    prevGray = gray;

    cv::Mat blurred = preprocess(gray);
    cv::Mat edges   = detectEdges(blurred);
    if (!motion.empty()) cv::bitwise_or(edges, motion, edges);

    cv::Mat col;
    cv::cvtColor(edges, col, cv::COLOR_GRAY2BGR);
    cv::Mat out;
    cv::hconcat(frame, col, out);
    return out;
}

//─── 6) EVALUATION LOOP ─────────────────────────────────────────────────────
int runEvaluation(const std::string& split) {
    std::string base = "archive";
    std::string imgF = base + "/images/" + split;
    std::string gtF  = base + "/converted_ground_truth/" + split;
    std::string sigma_string = std::to_string(sigma);
    std::ofstream log("bsds_" + split + "_autoCanny_" + sigma_string + ".csv");
    log << "image,P,R,F1,ms\n";
    double sumF=0;
    int n=0;
    for (auto& p : fs::directory_iterator(imgF)) {
        std::string name = p.path().stem().string();
        cv::Mat img = cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        n++;

        cv::Mat gtSkel = loadGTSkel(gtF, name);
        if (gtSkel.empty()) continue;

        cv::Mat blurred = preprocess(img);
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat det = detectEdges(blurred);
        auto t1 = std::chrono::high_resolution_clock::now();

        double P,R,F1;
        computePRF_dt(det, gtSkel, P,R,F1);
        sumF += F1;
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        log << name << "," << P << "," << R << "," << F1 << "," << ms << "\n";
    }
    log.close();
    std::cout << "Evaluated " << n << " images on '" << split << "' with auto-Canny.\n";
    std::cout << "Average F1: " << sumF/n << "\n";
    return 0;
}

//─── 7) MAIN ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    if (argc==2 && std::string(argv[1])=="--train")
        return runEvaluation("train");

    // video mode
    if (argc == 3 && std::string(argv[1]) == "--eval") {
        const std::string videoPath = argv[2];
        cv::VideoCapture cap(videoPath);
        printf("Opening video file: %s\n", videoPath.c_str());
        // cv::VideoCapture cap(videoFileName);
        if (!cap.isOpened()) {
            std::cerr<<"Cannot open video\n"; return -1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps<=0) fps=30;
        double delayMs = 1000.0 / fps;

        cv::Mat frame, prevGray;
        std::cout<<"Press 'q' to quit.\n";
        while (cap.read(frame)) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cv::Mat out = processFrame(frame, prevGray);
            cv::imshow("Original | AutoEdges", out);
            auto t1 = std::chrono::high_resolution_clock::now();
            double procMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            int delay = std::max(1, static_cast<int>(delayMs - procMs));
            if ((char)cv::waitKey(delay) == 'q') break;

        }
        return 0;
    }
    return -1;
}
