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

"""

in lots of models at config.py
Get recall value for for ovthresh in [0.05-0.95]


"""

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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
    # print forward_kwargs
    blobs_out = net.forward(**forward_kwargs)
    # print blobs_out

    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # print "rois: ", rois
    # print "rois.shape: ", rois.shape
    # pool5 = net.blobs['pool5'].data.copy()
    # print "roi_pool5: ", pool5
    # print "roi_pool5.shape: ", pool5.shape
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]
    # print "boxes: ", boxes
    # print "boxes.shape: ", boxes.shape

    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']
    # print "scores: ", scores

    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def processboxes(allboxes, imagenames):
    """

    输入：allboxes格式为：
        每一行代表[boxes, score]，总共有图片数这么多行
        allboxes[i] 表示第i张图片
    输出：allboxes格式为：
        每一行为：[图片编号，置信度， box]一行只对应一张图片的一个目标框
        如果一张图片有多个目标被检测出来，则放置多行，如：
        ['000002', '0.137', '74.3', '1.0', '335.0', '149.8']
        ['000002', '0.096', '10.7', '139.8', '224.8', '239.7']

    """
    boxes = []

    # 遍历allboxes进行处理

    for im_ind, index in enumerate(imagenames):
        dets = allboxes[im_ind]
        # 提取当前测试图片的对应种类目标的检测结果，如果为空则直接跳过下面过程
        if dets == []:
            continue
        # 不为空则进行放入目标boxes中
        # dets为[
        #    [69   232   100   296.47 0.071],
        #    [89   333   54.5  98.5   0.121]      
        # ] 这样的格式，二维矩阵
        for k in xrange(dets.shape[0]):
            boxes.append([index, dets[k, -1],
            dets[k, 0] + 1, dets[k, 1] + 1,
            dets[k, 2] + 1, dets[k, 3] + 1])
    return boxes

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
        # print "allboxes[j][i]: ", allboxes[j][i]
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
    # 1、求解出测试集的每个图片名，也就是解析test.txt，以备后用
    # 2、解析VOC2007的标注文件缓存，也就是读取所有的标注框
    # 3、遍历models，对每一个models执行操作：
    #   1）求出models的对检测集中每一张图片的检测结果，检测结果表示为一个目标一行，每个目标单独列出
    #   对于每个目标种类进行分别操作，for cls in classes:
    #         提取所有的测试集标注框，也就是所有的应该被检测到的目标，同样也是recall的分母，groundtrue
    #       １、取出当前测试集中所有的当前cls对应的测试集标注结果
    #       2、对于当前cls的检测结果分成3块，图片名、检测目标置信度、对应框位置，其中三块通过id也就是索引进行一一对应
    #       for ovthresh in [0.05-0.95]:
    #           1、将检测结果进行置信度由高到低进行排序，一一计算recall的两个值fp以及tp
    #           2、计算recall、precision以及ap值，并保存
    # 保存格式：
    # models_name train_dataset conv_net_name:zf/vgg16 class ovthresh recall precision ap


# √ 1、求解出测试集的每个图片名，也就是解析test.txt，以备后用 
    imagesetfile = osp.join(cfg.SOURCE_PATH, cfg.TEST_TXT)
    # print imagesetfile
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print imagenames
    # 遍历allboxes进行处理

# √ 生成我们要的overthresh区间：
    step = 0.05
    thresholds = np.arange(0.3, 0.95 + 1e-5, step)
    # print thresholds
    # input()
# √ 打开结果保存文件，追加形式打开，以免破坏原有数据：
    resultfile = open(osp.join(cfg.SOURCE_PATH, cfg.RES_TXT), "a")

#   3、进行目标检测，对模型进行遍历：
    for mod_ind, model in enumerate(cfg.MODELS):
        # 保存结果初始化：
        result = ""
        caffemodel = osp.join(cfg.SOURCE_PATH, model)
        # print caffemodel
        splitmodel = caffemodel.strip().split('/')
        # print splitmodel
        modelname = splitmodel[8]
        # print modelname
        traindataset = splitmodel[9]
        # print traindataset
        conv_net_name = splitmodel[10].split('_')[0]
        # print modelname
        # print traindataset
        # print conv_net_name
        # 获取网络结构参数
        prototxt = osp.join(cfg.SOURCE_PATH, cfg.PROTO_PATH, conv_net_name.upper(), modelname, "test.prototxt")
        # print "prototxt: ", prototxt
        # 获取测试集路径：
        testimagepath = osp.join(cfg.SOURCE_PATH, cfg.TEST_IMAGE)
        # 初始化caffe网络
        net = initmodel(prototxt, caffemodel)
        # input()
        # 开始使用初始化的caffe网络加上所给测试集数据 进行检测
        allboxes = detection(net, testimagepath, imagenames)

        # √ 2、解析VOC2007的标注文件缓存，也就是读取所有的标注框
        cachefile = osp.join(cfg.SOURCE_PATH, cfg.ANNOTS_PKL)
        # print cachefile
        # 读取缓存文件，存入recs中
        with open(cachefile, 'r') as f:
                recs = cPickle.load(f)
        # print "recs: ", recs
        # 对每一种目标进行遍历：
        for cls_ind, cls in enumerate(cfg.CLASSES):
            if cls == '__background__':
                continue
            # 找到所有当前类别目标的标注框

            # 此时返回的allboxes格式为：
            # 每一行代表[boxes, score]，总共有图片数×种类数
            # allboxes[j][i] 表示第i张图片，对于第j种种类的目标检测结果
            # 我们需要处理第cls_ind种类对应的检测结果allboxes[cls_ind]：
            # 每一行为：[图片编号，置信度， box]一行只对应一张图片的一个目标框
            # 如果一张图片有多个目标被检测出来，则放置多行，如：
            # ['000002', '0.137', '74.3', '1.0', '335.0', '149.8']
            # ['000002', '0.096', '10.7', '139.8', '224.8', '239.7']
            allboxes_cls = processboxes(allboxes[cls_ind], imagenames)

            # 取出第一列值，表明总共有多少个目标被检测出来
            image_ids = [x[0] for x in allboxes_cls]
            # print "image_ids: ", image_ids
            # 取出第二列值，表明每个被检测出来的目标的置信度
            confidence = np.array([float(x[1]) for x in allboxes_cls])
            # print "confidence: ", confidence
            # 取出第三到最后列的值，表明每个被检测出来的目标的候选框
            BB = np.array([[float(z) for z in x[2:]] for x in allboxes_cls])
            # print "BB: ", BB 

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            # print "sorted_ind: ", sorted_ind
            # print "sorted_scores: ", sorted_scores
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            for ovthresh in np.float64(cfg.OVERTHRESH[mod_ind]):
                print "ovthresh: ", ovthresh
                tp = np.zeros(nd)
                fp = np.zeros(nd)
                # print "nd: ", nd
                print "tp: ", tp
                print "fp: ", fp
                class_recs = {}
                npos = 0
                for imagename in imagenames:
                    R = [obj for obj in recs[imagename] if obj['name'] == cls]
                    bbox = np.array([x['bbox'] for x in R])
                    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                    det = [False] * len(R)
                    npos = npos + sum(~difficult)
                    class_recs[imagename] = {'bbox': bbox,
                                            'difficult': difficult,
                                            'det': det}
                # print "class_recs: ", class_recs
                for d in range(nd):
                    R = class_recs[image_ids[d]]
                    bb = BB[d, :].astype(float)
                    ovmax = -np.inf
                    BBGT = R['bbox'].astype(float)

                    if BBGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(BBGT[:, 0], bb[0])
                        iymin = np.maximum(BBGT[:, 1], bb[1])
                        ixmax = np.minimum(BBGT[:, 2], bb[2])
                        iymax = np.minimum(BBGT[:, 3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        if not R['difficult'][jmax]:
                            if not R['det'][jmax]:
                                tp[d] = 1.
                                R['det'][jmax] = 1
                            else:
                                fp[d] = 1.
                    else:
                        fp[d] = 1.

                print "fp: ", fp
                print "tp: ", tp 
                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(npos)
                print "fp: ", fp
                print "tp: ", tp 
                print "rec: ", rec
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = voc_ap(rec, prec, False)
                print "prec: ", prec
                print "ap: ", ap
                # 保存格式：
                # models_name train_dataset conv_net_name:zf/vgg16 class ovthresh recall precision ap
                # print "write result to ", resultfile
                # print "modelname: ", modelname, type(modelname)
                # print "conv_net_name: ", conv_net_name, type(conv_net_name)
                # print "cls: ", cls, type(cls)
                # print "ovthresh: ", ovthresh, type(ovthresh)
                # print "rec: ", rec, type(rec)
                # print "rec[-1]: ", rec[-1], type(rec[-1])
                # print "prec: ", prec, type(prec)
                # print "ap: ", ap, type(ap)
                recall = round(float(rec[-1]), 4)
                preci = round(float(prec[-1]), 4)
                apf = round(float(ap), 4)
                result = modelname + " " + traindataset + " "+ conv_net_name + " " + cls + " " + str(ovthresh) + " " + str(recall) + " " + str(preci) + " " + str(apf) + "\n"
                resultfile.write(result)
    resultfile.write("\n\n\n")
    resultfile.close()