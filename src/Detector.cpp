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

std::vector<sl::uint2> Detector::cvt(const cv::Rect &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

std::vector<sl::CustomBoxObjectData> Detector::detect(cv::Mat &rgb_mat)
{
    // Prepare the Input Data for the Engine
    // letterbox BGR to RGB
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

    std::vector<sl::CustomBoxObjectData> objects_in;
    for (auto &it : res) {
        sl::CustomBoxObjectData tmp;
        cv::Rect r = get_rect(rgb_mat, it.bbox);
        // Fill the detections into the ZED SDK format
        tmp.unique_object_id = sl::generate_unique_id();
        tmp.probability = it.conf;
        tmp.label = (int) it.class_id;
        tmp.is_grounded = ((int) it.class_id == 0);
        tmp.bounding_box_2d = cvt(r);
        /**
        * 0 ---- 1
        * |      |
        * 3 -----2
        */
        /**
            float size_y = (tmp.bounding_box_2d[3][1]-tmp.bounding_box_2d[0][1]);
            float size_x = (tmp.bounding_box_2d[1][0]-tmp.bounding_box_2d[0][0]);

            if (size_y < 500.0 && size_x < 500.0) {
            	objects_in.push_back(tmp);
            }
        else
        	std::cout << "Object too big to be real" << std::endl;
        **/

        objects_in.push_back(tmp);
    }

    return objects_in;
}
