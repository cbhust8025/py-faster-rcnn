#!/usr/bin/env python
# -*- coding=utf-8 -*-

# --------------------------------------------------------
# config.py
# Copyright (c) 2017 CB
# Written by Leal Cheng
# --------------------------------------------------------

# begin to import constant
# import os
import os.path as osp
import time
# end to import constant

# begin to import thirdpart
import cPickle
import numpy as np
import caffe
import cv2
# end to import thirdpart

# begin to import my custom
import _init_paths
from fast_rcnn.config import cfg
from utils.timer import Timer
from fast_rcnn.test import im_detect
from utils.blob import im_list_to_blob
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
# end to import my custom

# export PYTHONPATH=/home/bcheng/disk/new-py-faster-rcnn/py-faster-rcnn/caffe-fast-rcnn/python/:$PYTHONPATH
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)
    # print "im_detect blobs: ", blobs
    # print "im_detect im_scales: ", im_scales
    # input()
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    # print "cfg.TEST.HAS_RPN: ", cfg.TEST.HAS_RPN
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]

    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']

    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def detection(net, testimagepath, imagenames):
    """

    输入：caffe网络实例化对象、测试集路径、测试集每张图片编号
    输出：所有的检测结果，每一行代表[boxes, score]，总共有图片数×种类数
        allboxes[j][i] 表示第i张图片，对于第j种种类的目标检测结果

    """
    allboxes = [[[] for _ in xrange(len(imagenames))]
                 for _ in xrange(len(cfg.CLASSES))]
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    thresh = cfg.MIN_THRESH
    max_per_image = cfg.MAX_PER_IMAGE
    # 遍历每一张测试集图片
    for i in xrange(len(imagenames)):
        box_proposals = None
        #读取要检测的测试集图片
        imagename = osp.join(testimagepath, imagenames[i]) + cfg.IMAGE_TYPE
        # print imagename
        # input() 
        im = cv2.imread(imagename)
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        # print "scores: ", scores
        # print "scores.shape: ", scores.shape
        # print "boxes: ", boxes
        # print "boxes.shape: ", boxes.shape
        _t['im_detect'].toc()
        # input()
        _t['misc'].tic()
        # 对于当前图片的检测结果，遍历每一个目标种类，分别存入allboxes中：
        # 跳过j= 0，因为j=0对应背景这一种，不进行检测
        for j in xrange(1, len(cfg.CLASSES)):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.MYNMS)
            cls_dets = cls_dets[keep, :]
            allboxes[j][i] = cls_dets
        # Limit to max_per_image detections *over all classes*
        # 对每张照片的结果进行限制，取最大的100张
        if max_per_image > 0:
            image_scores = np.hstack([allboxes[j][i][:, -1]
                                      for j in xrange(1, len(cfg.CLASSES))])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, len(cfg.CLASSES)):
                    keep = np.where(allboxes[j][i][:, -1] >= image_thresh)[0]
                    allboxes[j][i] = allboxes[j][i][keep, :]
        _t['misc'].toc()
        print "allboxes[j][i]: ", allboxes[j][i]
        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, len(imagenames), _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return allboxes

def initmodel(prototxt, model):
    """
    init 函数：
        1、初始化caffe网络，并返回网络实例对象
    """
    cfg.GPU_ID = 0
    cfg.TEST.HAS_RPN = True
    while not osp.exists(model):
        print('Waiting for {} to exist...'.format(model))
        time.sleep(10)
    
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    # init caffe net
    net = caffe.Net(prototxt, model, caffe.TEST)

    return net

if __name__ == '__main__':
    # √ 1、求解出测试集的每个图片名，也就是解析test.txt，以备后用 
    imagesetfile = osp.join(cfg.SOURCE_PATH, cfg.TEST_TXT)
    # print imagesetfile
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    modelfile = "/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/d_faster_rcnn_end2end/d_faster_rcnn_end2end_2x/voc_2007_trainval/vgg16_faster_rcnn_iter_80000.caffemodel"
    prototxt = "/home/cb/Documents/MyGitHub/py-faster-rcnn/models/pascal_voc/VGG16/d_faster_rcnn_end2end_2x/testold.prototxt"

    net = initmodel(prototxt, modelfile)
    # 获取测试集路径：
    testimagepath = osp.join(cfg.SOURCE_PATH, cfg.TEST_IMAGE)
    # 开始使用初始化的caffe网络加上所给测试集数据 进行检测
    allboxes = detection(net, testimagepath, imagenames)