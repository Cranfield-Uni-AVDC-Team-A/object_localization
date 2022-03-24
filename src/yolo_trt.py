#!/usr/bin/env python3
from __future__ import print_function

import ctypes
import time
import sys

import argparse
import rospkg
pack_path = rospkg.RosPack().get_path("yolo_ros")
sys.path.append(pack_path)

import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils.helpers import *
#from utils.helpers import *

class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def yolo_preprocess(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    resized = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    return img_in

def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i],
                         x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i],
                         y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def yolo_postprocess(trt_outputs, img_w, img_h, conf_th, nms_threshold=0.5):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold

    # Returns
        boxes, scores, classes (after NMS)
    """
    # concatenate outputs of all yolo layers
    detections = np.concatenate(
        [o.reshape(-1, 7) for o in trt_outputs], axis=0)

    # drop detections with score lower than conf_th
    box_scores = detections[:, 4] * detections[:, 6]
    pos = np.where(box_scores >= conf_th)
    detections = detections[pos]

    # scale x, y, w, h from [0, 1] to pixel values
    detections[:, 0] *= img_w
    detections[:, 1] *= img_h
    detections[:, 2] *= img_w
    detections[:, 3] *= img_h

    # NMS
    nms_detections = np.zeros((0, 7), dtype=detections.dtype)
    for class_id in set(detections[:, 5]):
        idxs = np.where(detections[:, 5] == class_id)
        cls_detections = detections[idxs]
        keep = _nms_boxes(cls_detections, nms_threshold)
        nms_detections = np.concatenate(
            [nms_detections, cls_detections[keep]], axis=0)
    if len(nms_detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0, 1), dtype=np.float32)
        classes = np.zeros((0, 1), dtype=np.float32)
    else:
        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
    

    return boxes, scores, classes

class Yolo_TRT(object):
    """Yolo_TRT class encapsulates everythings needed to run Yolo with Tensor RT Engine"""

    def get_engine(self):
        engine_path = '%s.trt' % self.model
        trt.init_libnvinfer_plugins(None, "") 
        print("Reading engine from file {}".format(engine_path))
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def __init__(self, model, input_shape, category_num=2, cuda_ctx=None):
        """Constructor :: Initialize the TensorRT plugins, engine and context."""
        self.model = model
        self.input_shape = input_shape
        self.category_number = category_num
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, 1)
        #print("[Yolo_TRT] Initialization Done ~")

    def __del__(self):
        """Destructor :: Free the CUDA memories"""
        del self.outputs
        del self.inputs
        del self.stream

    def allocate_buffers(self, engine, batch_size=1):
        """Allocate all host/device in/out buffers required by the TensorRT engine"""

        inputs = []
        outputs =[]
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)

            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1
            
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    

    def detect(self, img, conf_th=0.3):

        print("Now start detecting")
        # Start the timer
        ta = time.time()
        img_preprocessed = yolo_preprocess(img, self.input_shape)
        print("Shape of the network input: ", img_preprocessed.shape)
        
        trt_outputs = []
        self.inputs[0].host = img_preprocessed
        
        if self.cuda_ctx:
            self.cuda_ctx.push()
        
        context = self.context
        bindings = self.bindings
        inputs = self.inputs
        outputs = self.outputs
        stream = self.stream

        print('Length of inputs: ', len(inputs))
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()

        # Return only the host outputs.
        trt_outputs = [out.host for out in outputs]

        print('Len of outputs: ', len(trt_outputs))

        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, self.category_number)

        tb = time.time()
    
        print('-----------------------------------')
        print('    TRT inference time: %f' % (tb - ta))
        print('-----------------------------------')
        
        del context
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        #boxes, scores, classes = yolo_postprocess(trt_outputs, img.shape[1], img.shape[0], conf_th)
        detections = post_processing(0.4, 0.6, trt_outputs)
        """
        box[0]: x1
        box[1]: y1
        box[2]: x2
        box[3]: y2

        box[5]: class confidence
        box[6]: class ID
        """
        # clip x1, y1, x2, y2 within original image
        #boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        #boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)
        #print("num of detection = ")
        #print(len(detections[0]))
    
        return detections[0]


        












