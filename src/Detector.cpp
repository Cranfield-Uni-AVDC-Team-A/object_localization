#include "object_localization/Detector.h"

Detector::Detector(std::string engine_name, int input_w, int input_h, int num_classes, float yolo_thresh, float nms_thresh)
    : inputW_(input_w),
      inputH_(input_h),
      numClasses_(num_classes),
      yoloThresh_(yolo_thresh),
      nmsThresh_(nms_thresh)
{
    // Runtime
    runtime_ = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // Engine
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        //return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    assert(engine_ != nullptr);

    // Context
    context_ = engine_->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // Buffers
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex_ = engine_->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex_ == 0);
    assert(outputIndex_ == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex_], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof (float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex_], BATCH_SIZE * OUTPUT_SIZE * sizeof (float)));

    // Stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    assert(BATCH_SIZE == 1); // only support batch 1 for now

}

Detector::~Detector()
{
    // release the stream and the buffers
    cudaStreamDestroy(stream_);
    CUDA_CHECK(cudaFree(buffers[inputIndex_]));
    CUDA_CHECK(cudaFree(buffers[outputIndex_]));

    // Destroy the engine
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();

}


void Detector::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof (float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof (float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

}


std::vector<Detections2D> Detector::detect(cv_bridge::CvImagePtr rgb_image_ptr)
{
    // Prepare the Input Data for the Engine
    // letterbox BGR to RGB
    cv::Mat rgb_mat = rgb_image_ptr->image;
    cv::Mat pr_img = preprocess_img(rgb_mat, INPUT_W, INPUT_H);
    int i = 0;
    int batch = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[batch * 3 * INPUT_H * INPUT_W + i] = (float) uc_pixel[2] / 255.0;
            data[batch * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float) uc_pixel[1] / 255.0;
            data[batch * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // Running Inference
    doInference(*context_, stream_, buffers, data, prob, BATCH_SIZE);
    std::vector<std::vector < Yolo::Detection >> batch_res(BATCH_SIZE);
    auto& res = batch_res[batch];
    nms(res, &prob[batch * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    
    std::vector<Detections2D> detections_array;
    for (auto &it : res) {
        Detections2D detection;
        cv::Rect r = get_rect(rgb_mat, it.bbox);
        detection.rectangle_box = r; // top_left_x, top_left_y, width_of_bbox, height_of_bbox
        detection.classID = (int) it.class_id;
        detection.prob = it.conf;
        detections_array.push_back(detection);
    }

    return detections_array;
}
