#  Copyright (C) 2020 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ctypes import *
from typing import List
from pynq_dpu import DpuOverlay

import cv2
import numpy as np
import vart
import pathlib
import xir
import time
import math

from threading import Thread
from queue import Queue

from cpu_nms import nms

def letter_box(yolox, image, input_w, input_h):
    scale = min(input_w / image.shape[1], input_h / image.shape[0])
    new_w = int(round(image.shape[1] * scale))
    new_h = int(round(image.shape[0] * scale))
    image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = np.array(image, dtype=np.float32, order='C')

    pad_top = int((input_h - new_h)/ 2 + 0.5)
    pad_bottom = int((input_h - new_h)/ 2)
    pad_left = int((input_w - new_w)/ 2 + 0.5)
    pad_right = int((input_w - new_w)/ 2)
    new_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(0))
    return new_image, {'scale': scale, 'pad_top': pad_top, 'pad_bottom': pad_bottom, 'pad_left': pad_left, 'pad_right': pad_right}


def read_img(q, img_list):
    cnt = 0
    for path in img_list:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        q.put(img)
        cnt += 1

def pre(read_q, pre_q, yolox, img_num, shapeIn, output_num, shapeOuts, input_w, input_h):
    cnt = 0
    while cnt < img_num:
        img = read_q.get()
        if img is None:
            continue
        inputData = [np.empty(shapeIn, dtype=np.float32, order="C")]
        outputData = []
        for i in range(output_num):
            outputData.append(np.empty(shapeOuts[i], dtype=np.float32, order="C"))
        img, params = letter_box(yolox, img, input_w, input_h)
        imageRun = inputData[0]
        imageRun[0, ...] = img[..., np.newaxis]
        pre_q.put((inputData, outputData, params))
        cnt += 1

def run(yolox, pre_q, post_q, img_num):
    cnt = 0
    while cnt < img_num:
        ret = pre_q.get()
        if ret is None:
            continue
        # run dpu
        inputData, outputData, params = ret[0], ret[1], ret[2]
        job_id = yolox.runner.execute_async(inputData, outputData)
        yolox.runner.wait(job_id)
        post_q.put((outputData, params))
        cnt += 1

def post(yolox, post_q, img_num, result):
    cnt = 0
    while cnt < img_num:
        ret = post_q.get()
        if ret is None:
            continue
        outputData, params = ret[0], ret[1]
        ret = postprocess(yolox, outputData, params)
        result.append(ret)
        cnt += 1

def postprocess(yolox, mlvl_preds, params):
    conf_thresh_nosig1 = -4.5951 # 0.01
    strides=[8,16,32]
    mlvl_preds_list = []
    mlvl_preds_list.append([mlvl_preds[6],mlvl_preds[7],mlvl_preds[8]])
    mlvl_preds_list.append([mlvl_preds[3],mlvl_preds[4],mlvl_preds[5]])
    mlvl_preds_list.append([mlvl_preds[0],mlvl_preds[1],mlvl_preds[2]])
    mlvl_preds = []
    mlvl_shapes = []
    mlvl_strides = []
    for lvl_ix in range(3):
        b, h, w, _ = mlvl_preds_list[lvl_ix][2].shape
        k = h * w
        mlvl_shapes.append((h, w, k))
    mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]
    mlvl_location = yolox.get_anchors(mlvl_shapes)
    mlvl_locations = []

    for lvl_ix in range(3):
        p_cls = mlvl_preds_list[lvl_ix][0]
        p_loc = mlvl_preds_list[lvl_ix][1]
        p_obj = mlvl_preds_list[lvl_ix][2]
        m_obj = (p_obj > conf_thresh_nosig1).squeeze(-1)
        f_cls = p_cls[m_obj][None]
        f_loc = p_loc[m_obj][None]
        f_obj = p_obj[m_obj][None]
        f_strides = np.zeros((f_obj.shape[0], f_obj.shape[1], 2))
        f_strides.fill(strides[lvl_ix])
        f_locations = mlvl_location[lvl_ix][None][m_obj]
        mlvl_preds.append([f_cls, f_loc, f_obj])
        mlvl_strides.append(f_strides)
        mlvl_locations.append(f_locations)
    results = yolox.predict_nosig(mlvl_preds, mlvl_strides, mlvl_locations, params)
    return results

class YOLOX:
    def __init__(self, bit_name = "dpu.bit", model_name = "yolox.xmodel"):
        """
            get input tensor and get four output tensor
        """
        self.overlay = DpuOverlay(bit_name)
        self.overlay.load_model(model_name)
        self.runner = self.overlay.runner

        self.input_tensors = self.runner.get_input_tensors()
        self.input_dims = self.input_tensors[0].dims
        self.input_h = self.input_tensors[0].dims[1]
        self.input_w = self.input_tensors[0].dims[2]
        self.output_tensors = self.runner.get_output_tensors()
        self.block_size = 1
        self.num_classes = 80
        self.pre_nms_thresh = 0.01
        self.iou_thresh = 0.65
        self.top_n = 300

    def run_test(self, img_list):
        inputTensors = self.runner.get_input_tensors()
        outputTensors = self.runner.get_output_tensors()
        shapeIn = tuple(inputTensors[0].dims)
        output_num = len(outputTensors)
        shapeOuts = []
        input_w = inputTensors[0].dims[1] * self.block_size
        input_h = inputTensors[0].dims[2] * self.block_size
        for i in range(output_num):
            shapeOuts.append(tuple(outputTensors[i].dims))

        img_num = len(img_list)
        task_num = 2

        read_q = list()
        pre_q = list()
        post_q = list()

        read_t = list()
        pre_t = list()
        run_t = list()
        post_t = list()

        result = []
        img_num_per_task = int(img_num / task_num)
        for i in range(task_num):
            read_q.append(Queue(3))
            pre_q.append(Queue(3))
            post_q.append(Queue(3))
            start = i * img_num_per_task
            end = min((i + 1) * img_num_per_task, img_num)
            process_len = end - start
            read_t.append(Thread(target=read_img, args=(read_q[-1], img_list[start:end])))
            pre_t.append(Thread(target=pre, args=(read_q[-1], pre_q[-1], self, process_len, shapeIn, output_num, shapeOuts, input_w, input_h)))
            run_t.append(Thread(target=run, args=(self, pre_q[-1], post_q[-1], process_len)))
            res = []
            post_t.append(Thread(target=post, args=(self, post_q[-1], process_len, res)))
            result.append(res)

        for t in read_t:
            t.start()
        for t in pre_t:
            t.start()
        for t in run_t:
            t.start()
        for t in post_t:
            t.start()

        for t in read_t:
            t.join()
        for t in pre_t:
            t.join()
        for t in run_t:
            t.join()
        for t in post_t:
            t.join()

        return [r for rr in result for r in rr]

    def predict_nosig(self, preds, mlvl_strides, mlvl_locations, params):
        max_wh = 4096
        preds = [np.concatenate(p,axis=2) for p in preds]
        preds = np.concatenate(preds, axis=1)
        all_strides = np.concatenate(mlvl_strides, axis=1)
        all_locations = np.concatenate(mlvl_locations, axis=0)
        B = preds.shape[0]
        det_results = []
        rois_keep = []
        for b_ix in range(B):
            pred_per_img = preds[b_ix]
            obj_conf = pred_per_img[:, -1]
            strides = all_strides[b_ix]

            class_conf = np.max(pred_per_img[:, :self.num_classes], axis = 1)
            class_pred = np.argmax(pred_per_img[:, :self.num_classes], axis = 1)

            det_scores_conf = self.sigmoid(class_conf) * self.sigmoid(obj_conf)
            conf_mask_keep = det_scores_conf > self.pre_nms_thresh

            # decode
            loc_pred = pred_per_img[:, self.num_classes:-1][conf_mask_keep]
            strides = strides[conf_mask_keep]
            locations = all_locations[conf_mask_keep]
            loc_pred[:, :2] *= strides
            loc_pred[:, :2] += locations
            loc_pred[:, 2:4] = np.exp(loc_pred[:, 2:4]) * strides

            detections = np.concatenate((loc_pred, det_scores_conf[:, np.newaxis][conf_mask_keep], class_pred[:, np.newaxis][conf_mask_keep].astype(np.float32)), axis=1)

            classes = detections[:,-1]
            boxes = detections[:,:4]
            boxes[:,0] = boxes[:,0] - boxes[:,2]/2
            boxes[:,1] = boxes[:,1] - boxes[:,3]/2
            boxes[:,2] = boxes[:,0] + boxes[:,2]
            boxes[:,3] = boxes[:,1] + boxes[:,3]
            boxes_hash = boxes + classes[:, None] * max_wh
            scores = detections[:,4:5]
            dets = np.hstack((boxes_hash, scores))
            idx_keep = nms(dets,self.iou_thresh)
            dets_keep = boxes[idx_keep]
            scores_keep = scores[idx_keep]
            classes_keep = classes[idx_keep]
            rois_keep = np.hstack((dets_keep, scores_keep, classes_keep[:, None]))

            if len(rois_keep)<=0:
                continue
            n = rois_keep.shape[0]
            if n > self.top_n:
                rois_keep = rois_keep[:self.top_n]
            det_results.append(np.concatenate([np.full((rois_keep.shape[0],1),b_ix),rois_keep],axis = 1))

        if len(det_results)<=0:
            return None
        bboxes = np.concatenate(det_results, axis = 1)
        bboxes[:,1]= bboxes[:,1] - params['pad_left']
        bboxes[:,2]= bboxes[:,2] - params['pad_top']
        bboxes[:,3]= bboxes[:,3] - params['pad_right']
        bboxes[:,4]= bboxes[:,4] - params['pad_bottom']
        bboxes[:,1:5] = bboxes[:,1:5] / params['scale']
        return bboxes

    def get_anchors(self, featmap_shapes):
        locations = []
        for level, shape in enumerate(featmap_shapes):
            h,w,_,stride = shape
            locations_per_lever = self.compute_locations_per_lever(h, w, stride)
            locations.append(locations_per_lever)
        return locations

    def compute_locations_per_lever(self, h, w, stride):
        shifts_x = np.arange(0, w*stride, step=stride, dtype=np.float32)
        shifts_y = np.arange(0, h*stride, step=stride, dtype=np.float32)
        shift_x, shift_y = np.meshgrid(shifts_y,shifts_x)
        locations = np.stack((shift_x, shift_y),axis=2)
        return locations

    def sigmoid(self, x):
       return 1 / (1 + np.exp(-x))



class Processor:
    def __init__(self, bit_name = "dpu.bit", model_name = "yolov3_tf_ultra96v2-b1600.xmodel"):
        self.processor = YOLOX(bit_name, model_name)
    def run(self, images):
        cv2.setUseOptimized(True)

        list_image = images
        result = self.processor.run_test(list_image)
        return result

if __name__ == "__main__":
    import sys
    detector = YOLOX("dpu.bit", sys.argv[1])
    team_name = 'spring'
    import lpcv_eval
    team = lpcv_eval.Team(team_name, batch_size = 1)

    import os
    images_list = os.listdir('images')
    images_list = ['images/'+path for path in images_list]


    import pynq
    rails = pynq.get_rails()

    import time
    os.system('echo 1 > /proc/sys/vm/drop_caches')
    t1 = time.time() * 1000
    recorder = pynq.DataRecorder(rails["PSINT_FP"].power)
    with recorder.record(0.05):
        results = detector.run_test(images_list)
    t2 = time.time() * 1000
    total_time = (t2 - t1) / 1000.
    energy = recorder.frame["PSINT_FP_power"].mean() * total_time

    total_energy = energy
    fps = 1000. / (total_time / 20288. * 1000)
    print(f'Time: {total_time} s, Energy: {total_energy} J, Latency: {total_time * 1000. / 20288} ms, FPS: {fps}')

    all_results = results
    save_results = []
    mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    for i in range(len(all_results)):
        if all_results[i] is None:
            continue
        for j in range(len(all_results[i])):
            label = mapping[int(all_results[i][j][6])]
            score = all_results[i][j][5]
            x = all_results[i][j][1]
            y = all_results[i][j][2]
            width = all_results[i][j][3] - all_results[i][j][1]
            height = all_results[i][j][4] - all_results[i][j][2]
            image_id = images_list[i].split('/')[-1].split('.')[0]
            save_results.append([image_id, x, y, width, height, label, score])
    team.save_results_json(save_results)
