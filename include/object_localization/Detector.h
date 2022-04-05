#pragma once
#include <iostream>
#include <chrono>
#include <cmath>
#include "object_localization/cuda_utils.h"
#include "object_localization/logging.h"
#include "object_localization/common.hpp"
#include "object_localization/utils.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof (Yolo::Detection) / sizeof (float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
inline const char* INPUT_BLOB_NAME = "data";
inline const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

struct Detections2D{
    cv::Rect rectangle_box; // (top_left_x, top_left_y, width_of_bbox, height_of_bbox)
    int classID;
    float prob;
};


class Detector{

public:
    Detector(std::string engine_name, int input_w, int input_h, int num_classes, float yolo_thresh, float nms_thresh);
    ~Detector();
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    //std::vector<Detections2D> detect(cv::Mat& rgb_mat);
    std::vector<Detections2D> detect(cv_bridge::CvImagePtr rgb_image_ptr);

private:
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    void* buffers[2];
    
    int inputIndex_;
    int outputIndex_;

    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];  // input
    float prob[BATCH_SIZE * OUTPUT_SIZE];;  // output
    int inputW_;
    int inputH_;
    int numClasses_;
    float yoloThresh_;
    float nmsThresh_;

};
