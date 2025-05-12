#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "EdgeDetector.hpp"
#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>

Image::Image(int w, int h) : width(w), height(h), data(w * h) {}

uint8_t& Image::at(int x, int y) {
    return data[y * width + x];
}

const uint8_t& Image::at(int x, int y) const {
    return data[y * width + x];
}

Image Image::fromMat(const cv::Mat& mat) {
    int w = mat.cols, h = mat.rows;
    Image img(w, h);

    if (mat.type() == CV_8UC1) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                img.at(x, y) = mat.at<uint8_t>(y, x);
            }
        }
    } else if (mat.type() == CV_8UC3) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                cv::Vec3b pixel = mat.at<cv::Vec3b>(y, x);
                img.at(x, y) = static_cast<uint8_t>(
                    0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            }
        }
    } else {
        throw std::runtime_error("Unsupported image type in fromMat()");
    }

    return img;
}


cv::Mat Image::toMat() const {
    cv::Mat mat(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            mat.at<uint8_t>(y, x) = at(x, y);
    return mat;
}

// Image EdgeDetector::process(const Image& input) {
//     Image gray = toGrayscale(input);
//     Image claheImg = applyCLAHE(gray);  // Apply CLAHE
//     Image blurred = gaussianBlur(claheImg);  // Apply Gaussian Blur after CLAHE
//     std::vector<float> magnitude, direction;
//     computeGradients(blurred, magnitude, direction);
//     Image suppressed = nonMaxSuppression(magnitude, direction, input.width, input.height);
//     Image finalEdges = doubleThresholdAndHysteresis(suppressed, 70.0f, 180.0f);
//     return finalEdges;
// }

Image EdgeDetector::process(const Image& input) {
    Image gray = toGrayscale(input);
    Image claheImg = applyCLAHE(gray);
    Image blurred = gaussianBlur(claheImg);

    // Compute dynamic thresholds
    cv::Mat blurredMat = blurred.toMat();
    double m = cv::mean(blurredMat)[0];
    double lower = std::max(0.0, 0.66 * m);
    double upper = std::min(255.0, 1.33 * m);

    std::vector<float> magnitude, direction;
    computeGradients(blurred, magnitude, direction);
    Image suppressed = nonMaxSuppression(magnitude, direction, input.width, input.height);
    Image edges = doubleThresholdAndHysteresis(suppressed, static_cast<float>(lower), static_cast<float>(upper));

    // Convert edges to cv::Mat
    cv::Mat edgesMat = edges.toMat();

    // Prepare motion mask
    cv::Mat currGray = claheImg.toMat(); // Already equalized grayscale
    cv::Mat motionMask;

    if (!firstFrame) {
        cv::absdiff(currGray, prevGray, motionMask);
        cv::threshold(motionMask, motionMask, 30, 255, cv::THRESH_BINARY);
    }

    // Save current frame
    currGray.copyTo(prevGray);

    // Temporal smoothing
    cv::Mat edgesF;
    edgesMat.convertTo(edgesF, CV_32F);

    if (firstFrame) {
        edgesF.copyTo(smoothedEdges);
        firstFrame = false;
    } else {
        cv::accumulateWeighted(edgesF, smoothedEdges, alpha);
    }

    // Convert smoothed edges back
    cv::Mat smoothed8U;
    smoothedEdges.convertTo(smoothed8U, CV_8U);

    // Combine with motion mask
    if (!motionMask.empty()) {
        cv::bitwise_or(smoothed8U, motionMask, smoothed8U);
    }

    return Image::fromMat(smoothed8U);
}


Image EdgeDetector::toGrayscale(const Image& input) {
    return input; // already grayscale
}

Image EdgeDetector::applyCLAHE(const Image& input) {
    cv::Mat gray = input.toMat();
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat equalized;
    clahe->apply(gray, equalized);
    Image result = Image::fromMat(equalized);
    return result;
}

Image EdgeDetector::gaussianBlur(const Image& input) {
    Image output(input.width, input.height);
    float kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int y = 1; y < input.height - 1; ++y) {
        for (int x = 1; x < input.width - 1; ++x) {
            float sum = 0;
            float weight = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    float w = kernel[ky + 1][kx + 1];
                    sum += w * input.at(x + kx, y + ky);
                    weight += w;
                }
            }
            output.at(x, y) = static_cast<uint8_t>(sum / weight);
        }
    }
    return output;
}

void EdgeDetector::computeGradients(const Image& input, std::vector<float>& magnitude, std::vector<float>& direction) {
    int w = input.width;
    int h = input.height;
    magnitude.resize(w * h);
    direction.resize(w * h);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    gx += Gx[ky + 1][kx + 1] * input.at(x + kx, y + ky);
                    gy += Gy[ky + 1][kx + 1] * input.at(x + kx, y + ky);
                }
            }
            int idx = y * w + x;
            magnitude[idx] = static_cast<float>(std::hypot(gx, gy));
            direction[idx] = static_cast<float>(std::atan2(gy, gx) * 180.0f / M_PI);
        }
    }
}

Image EdgeDetector::nonMaxSuppression(const std::vector<float>& magnitude, const std::vector<float>& direction, int width, int height) {
    Image output(width, height);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float angle = direction[idx];
            float mag = magnitude[idx];

            angle = static_cast<float>(std::fmod(angle + 180.0f, 180.0f));  // Normalize angle to [0, 180)

            float neighbor1 = 0, neighbor2 = 0;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                neighbor1 = magnitude[idx - 1];
                neighbor2 = magnitude[idx + 1];
            } else if (angle >= 22.5 && angle < 67.5) {
                neighbor1 = magnitude[(y - 1) * width + (x + 1)];
                neighbor2 = magnitude[(y + 1) * width + (x - 1)];
            } else if (angle >= 67.5 && angle < 112.5) {
                neighbor1 = magnitude[(y - 1) * width + x];
                neighbor2 = magnitude[(y + 1) * width + x];
            } else if (angle >= 112.5 && angle < 157.5) {
                neighbor1 = magnitude[(y - 1) * width + (x - 1)];
                neighbor2 = magnitude[(y + 1) * width + (x + 1)];
            }

            if (mag >= neighbor1 && mag >= neighbor2) {
                output.at(x, y) = static_cast<uint8_t>(std::min(255.0f, mag));
            } else {
                output.at(x, y) = 0;
            }
        }
    }

    return output;
}

Image EdgeDetector::doubleThresholdAndHysteresis(Image& input, float lowThresh, float highThresh) {
    int w = input.width;
    int h = input.height;
    Image output(w, h);

    std::queue<std::pair<int, int>> q;

    // First pass: classify strong and weak edges
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            if (input.at(x, y) >= highThresh) {
                output.at(x, y) = 255; // Strong edge
                q.push({x, y});
            } else if (input.at(x, y) >= lowThresh) {
                output.at(x, y) = 128; // Weak edge
            } else {
                output.at(x, y) = 0;   // Non-edge
            }
        }
    }

    // Second pass: apply hysteresis
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h && output.at(nx, ny) == 128) {
                    output.at(nx, ny) = 255;  // Convert weak edge to strong
                    q.push({nx, ny});
                }
            }
        }
    }

    return output;
}
