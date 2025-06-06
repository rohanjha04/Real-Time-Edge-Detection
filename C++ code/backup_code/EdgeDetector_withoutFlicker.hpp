#pragma once
#include <vector>
#include <opencv2/core.hpp>

struct Image {
    int width, height;
    std::vector<uint8_t> data;

    Image(int w, int h);
    uint8_t& at(int x, int y);
    const uint8_t& at(int x, int y) const;

    static Image fromMat(const cv::Mat& mat);
    cv::Mat toMat() const;
};

class EdgeDetector {
public:
    Image process(const Image& input);

private:
    Image toGrayscale(const Image& input);
    Image gaussianBlur(const Image& input);
    void computeGradients(const Image& input, std::vector<float>& magnitude, std::vector<float>& direction);
    Image nonMaxSuppression(const std::vector<float>& magnitude, const std::vector<float>& direction, int width, int height);
    Image doubleThresholdAndHysteresis(Image& input, float lowThresh, float highThresh);
};
