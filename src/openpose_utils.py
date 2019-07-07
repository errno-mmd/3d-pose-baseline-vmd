#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# openpose_utils.py

import os
import json
import re
import numpy as np
from collections import Counter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_invisible_leg(xyc):
    x = [0.0 for i in range(18)]
    y = [0.0 for i in range(18)]
    c = [0.0 for i in range(18)]
    for i in range(18):
        x[i] = xyc[i*3+0]
        y[i] = xyc[i*3+1]
        c[i] = xyc[i*3+2]

    spine_l_len = np.sqrt((x[1] - x[11])**2 + (y[1] - y[11])**2) if c[1] > 0 and c[11] > 0 else 0.0
    spine_r_len = np.sqrt((x[1] - x[8])**2 + (y[1] - y[8])**2) if c[1] > 0 and c[8] > 0 else 0.0
    clavicle_l_len = np.sqrt((x[1] - x[5])**2 + (y[1] - y[5])**2) if c[1] > 0 and c[5] > 0 else 0.0
    clavicle_r_len = np.sqrt((x[1] - x[2])**2 + (y[1] - y[2])**2) if c[1] > 0 and c[2] > 0 else 0.0
    if spine_l_len > 0 or spine_r_len > 0:
        spine_len = max(spine_l_len, spine_r_len)
    else:
        spine_len = max(clavicle_l_len * 4, clavicle_r_len * 4)
    if spine_len == 0.0:
        return

    th_conf = 0.1
    conf = 0.1

    # LHip
    if c[11] < th_conf:
        x[11] = (x[1] + x[5]) / 2
        y[11] = y[1] + spine_len
        c[11] = conf
    # LKnee
    if c[12] < th_conf:
        x[12] = x[11]
        y[12] = y[11] + spine_len * 0.8
        c[12] = conf
    # LAnkle
    if c[13] < th_conf:
        x[13] = x[12]
        y[13] = y[12] + spine_len * 0.8
        c[13] = conf
    # RHip
    if c[8] < th_conf:
        x[8] = (x[1] + x[2]) / 2
        y[8] = y[1] + spine_len
        c[8] = conf
    # RKnee
    if c[9] < th_conf:
        x[9] = x[8]
        y[9] = y[8] + spine_len * 0.8
        c[9] = conf
    # RAnkle
    if c[10] < th_conf:
        x[10] = x[9]
        y[10] = y[9] + spine_len * 0.8
        c[10] = conf

    for i in range(18):
        xyc[i*3+0] = x[i]
        xyc[i*3+1] = y[i]
        xyc[i*3+2] = c[i]


def read_openpose_json(openpose_output_dir, idx, is_debug=False, add_leg=False):
    if is_debug == True:
        logger.setLevel(logging.DEBUG)

    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    logger.info("start reading data: %s", openpose_output_dir)
    #load json files
    json_files = os.listdir(openpose_output_dir)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])
    cache = {}
    cache_confidence = {}
    smoothed = {}
    _past_tmp_points = []
    _past_tmp_data = []
    _tmp_data = []
    ### extract x,y and ignore confidence score
    is_started = False
    start_frame_index = 0
    end_frame_index = 0
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))

        if add_leg == True:
            add_invisible_leg(data["people"][idx]["pose_keypoints_2d"])

        # 12桁の数字文字列から、フレームINDEX取得
        frame_idx = int(re.findall(r"(\d{12})", file_name)[0])
        
        if frame_idx <= 0 or is_started == False:
            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame_index = frame_idx

        # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
        _data = data["people"][idx]["pose_keypoints_2d"]

        xy = []
        confidence = []
        for o in range(0,len(_data),3):
            xy.append(_data[o])
            xy.append(_data[o+1])
            confidence.append(_data[o+2])

        logger.debug("found {0} for frame {1}".format(xy, str(frame_idx)))
        #add xy to frame
        cache[frame_idx] = xy
        cache_confidence[frame_idx] = confidence
        end_frame_index = frame_idx

    # plt.figure(1)
    # drop_curves_plot = show_anim_curves(cache, plt)
    # pngName = '{0}/dirty_plot.png'.format(subdir)
    # drop_curves_plot.savefig(pngName)

    # # exit if no smoothing
    # if not smooth:
    #     # return frames cache incl. 18 joints (x,y)
    #     return cache

    if len(json_files) == 1:
        logger.info("found single json file")
        # return frames cache incl. 18 joints (x,y) on single image\json
        return cache

    if len(json_files) <= 8:
        raise Exception("need more frames, min 9 frames/json files for smoothing!!!")

    logger.info("start smoothing")

    # last frame of data
    last_frame = [start_frame_index-1 for i in range(18)]

    # threshold of confidence
    confidence_th = [0.3 for i in range(18)]
    for i in range(18):
        confidence_max = 0.0
        for j in range(len(json_files)):
            if cache_confidence[j][i] > confidence_max:
                confidence_max = cache_confidence[j][i]
        if confidence_max < 0.6:
            confidence_th[i] = confidence_max / 2

    ### smooth by median value, n frames 
    for frame, xy in cache.items():
        confidence = cache_confidence[frame]

        # joints x,y array
        _len = len(xy) # 36

        # frame range
        smooth_n = 7 # odd number
        one_side_n = int((smooth_n - 1)/2)
        one_side_n = min([one_side_n, frame-start_frame_index, end_frame_index-frame])

        smooth_start_frame = frame - one_side_n
        smooth_end_frame = frame + one_side_n

        # build frame range vector 
        frames_joint_median = [0 for i in range(_len)]
        # more info about mapping in src/data_utils.py
        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36 
        for x in range(0,_len,2):
            # set x and y
            y = x+1
            joint_no = int(x / 2)

            x_v = []
            y_v = []
            for neighbor in range(smooth_start_frame, smooth_end_frame + 1):
                if cache_confidence[neighbor][joint_no] >= confidence_th[joint_no]:
                    x_v.append(cache[neighbor][x])
                    y_v.append(cache[neighbor][y])

            if len(x_v) >= 1:
                # 配列の長さを奇数にする
                if len(x_v) % 2 == 0:
                    # x_v.append(cache[frame][x])
                    # y_v.append(cache[frame][y])
                    x_v.append(np.mean(x_v))
                    y_v.append(np.mean(y_v))

                # get median of vector
                x_med = np.median(sorted(x_v))
                y_med = np.median(sorted(y_v))

                # 前のフレームが欠損している場合、今回のフレームのデータと前回最後に取得できたデータで線形補間する
                if last_frame[joint_no] != frame -1:
                    if last_frame[joint_no] < start_frame_index:
                        # 過去に一度もデータを取得できなかった場合
                        last_value_x = x_med
                        last_value_y = y_med
                    else:
                        last_value_x = smoothed[last_frame[joint_no] - start_frame_index][x]
                        last_value_y = smoothed[last_frame[joint_no] - start_frame_index][y]

                    for frame_linear_in in range(last_frame[joint_no] + 1, frame):
                        # smoothed[frame_linear_in - start_frame_index][x] = last_value_x + (x_med - last_value_x) * (frame_linear_in - last_frame[joint_no]) / (frame - last_frame[joint_no])
                        # smoothed[frame_linear_in - start_frame_index][y] = last_value_y + (y_med - last_value_y) * (frame_linear_in - last_frame[joint_no]) / (frame - last_frame[joint_no])
                        smoothed[frame_linear_in - start_frame_index][x] = last_value_x
                        smoothed[frame_linear_in - start_frame_index][y] = last_value_y

                last_frame[joint_no] = frame

            else:
                # holding frame drops for joint
                # allow fix from first frame
                if frame > start_frame_index:
                    # get x from last frame
                    # logger.info("frame %s, x %s", frame, x)
                    x_med = smoothed[frame - start_frame_index -1][x]
                    # get y from last frame
                    y_med = smoothed[frame - start_frame_index -1][y]
                else:
                    x_med = 0
                    y_med = 0

            # logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
            # logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

            # build new array of joint x and y value
            frames_joint_median[x] = x_med 
            frames_joint_median[x+1] = y_med 

        smoothed[frame - start_frame_index] = frames_joint_median

    for n, (frame, xy) in enumerate(smoothed.items()):
        # 顔と耳のX位置を補正する
        joints = [(16, 1, 17, 17), (17, 1, 16, 16), (0, 1, 16, 17)]
        for (fromj, tojx, tojy1, tojy2) in joints:
            if cache_confidence[frame + start_frame_index][fromj] < confidence_th:
                # Fromが取れてない場合、Toから引っ張ってくる
                smoothed[frame][fromj*2] = smoothed[frame][tojx*2]
                smoothed[frame][fromj*2+1] = (smoothed[frame][tojy1*2+1] + smoothed[frame][tojy2*2+1]) / 2

    # return frames cache incl. smooth 18 joints (x,y)
    return start_frame_index, smoothed
