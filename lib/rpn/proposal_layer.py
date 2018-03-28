# -*- coding: utf-8-*- 
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from os import system

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        print "ProposalLayer setup function."
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        print "ProposalLayer layer_params: ", layer_params
        print "ProposalLayer bottom: ", bottom
        print "ProposalLayer len(bottom): ", len(bottom)
        print "ProposalLayer top: ", top
        print "ProposalLayer len(top): ", len(top)
        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        print "ProposalLayer self._feat_stride: ", self._feat_stride
        print "ProposalLayer anchor_scales: ", anchor_scales
        print "ProposalLayer self._anchors: ", self._anchors
        print "ProposalLayer self._num_anchors: ", self._num_anchors
        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        '''
        Algorithm:
        
        for each (H, W) location i
          generate A anchor boxes centered on cell i
          apply predicted bbox deltas at cell i to each of the A anchors
        clip predicted boxes to image
        remove predicted boxes with either height or width < threshold
        sort all (proposal, score) pairs by score from highest to lowest
        take top pre_nms_topN proposals before NMS
        apply NMS with threshold 0.7 to remaining proposals
        take after_nms_topN proposals after NMS
        return the top proposals (-> RoIs top, scores top)

        Algorithm(中文版)：

        1、对每一个（坐标i（ix,iy），长H=51,宽W=39）的点---特殊图上面的点：
             产生A个锚窗（此处A = 9），以(ix,iy)为中心
        2、将产生的每个锚窗映射到原图上，也就是特征图上面的最大不超过3*3的框映射到800*600的原图上。
        3、将超出原图尺寸的锚窗移除
        4、将所有（预测区域，预测分数）进行从高到底排序，选取前 RPN_PRE_NMS_TOP_N（此处等于12000）个
        5、将选取的12000个应用NMS算法，选取剩余的前2000个
        6、将剩余的2000个roi区域，前面添加一列0变成2000*5的矩阵传入下一层roi-data层。
        '''
        # proposal层只接受单个的item batch
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'
        print "ProposalLayer forward function."
        print "ProposalLayer bottom: ", bottom
        print "ProposalLayer top: ", top
        
        # 读取配置文件中的各项参数
        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE
        print "ProposalLayer cfg_key: ", cfg_key
        print "ProposalLayer pre_nms_topN: ", pre_nms_topN
        print "ProposalLayer post_nms_topN: ", post_nms_topN
        print "ProposalLayer nms_thresh: ", nms_thresh
        print "ProposalLayer min_size: ", min_size

        # 取出下面三层传入的数据，分别是
            # bottom: 'rpn_cls_prob_reshape'   传入打分数据   bottom[0]
            # bottom: 'rpn_bbox_pred'          传入每个打分对应的框 bottom[1]，位于特征图上
            # bottom: 'im_info'                传入原图的尺寸，便于将框放缩回原图 bottom[2]
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        # ProposalLayer scores.shape:  (1, 9, 51, 39)
        # ProposalLayer bbox_deltas.shape:  (1, 36, 51, 39)
        # ProposalLayer im_info:  [ 800.，600.，1.60000002]
        print "ProposalLayer scores: ", scores
        print "ProposalLayer scores.shape: ", scores.shape
        print "ProposalLayer bbox_deltas: ", bbox_deltas
        print "ProposalLayer bbox_deltas.shape: ", bbox_deltas.shape
        print "ProposalLayer im_info: ", im_info


        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 找到总共有多少个打分
        # ProposalLayer height:  51
        # ProposalLayer width:  39
        # 打分个数 = height * width * A(锚窗个数)
        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        print "ProposalLayer height: ", height
        print "ProposalLayer width: ", width

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # 特征图 × 16 = 原图， 映射回去
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        print "ProposalLayer shift_x: ", shift_x
        print "ProposalLayer shift_y: ", shift_y
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        print "ProposalLayer shift_x(meshgrid): ", shift_x
        print "ProposalLayer shift_y(meshgrid): ", shift_y
        print "ProposalLayer shifts: ", shifts
        print "ProposalLayer shifts.shape: ", shifts.shape

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # 将产生的A个锚窗附加到每个滑窗点上，将会产生K*A个roi区域
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        print "ProposalLayer anchors: ", anchors
        print "ProposalLayer anchors.shape: ", anchors.shape
        anchors = anchors.reshape((K * A, 4))

        print "ProposalLayer A: ", A
        print "ProposalLayer K: ", K
        print "ProposalLayer anchors（reshape）: ", anchors
        print "ProposalLayer anchors（reshape）.shape: ", anchors.shape
        # ProposalLayer anchors.shape:  (1989, 9, 4) 1989 = 51 * 39 
        # ProposalLayer A:  9
        # ProposalLayer K:  1989
        # ProposalLayer anchors（reshape）:  (17901, 4)
        # 共产生17901=1989*9个兴趣区域，每个区域用1*4来表示
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # 将bottom[1]中传入的区域（共 36/4 × 51 × 39个区域）矩阵（1,36,51,39）
        # 转换成（17901， 4）矩阵
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        print "ProposalLayer bbox_deltas: ", bbox_deltas
        print "ProposalLayer bbox_deltas.shape: ", bbox_deltas.shape
        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        # 将bottom[0]中的打分（1，9，51,39）矩阵转换成（17901,1） 矩阵
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        print "ProposalLayer scores: ", scores
        print "ProposalLayer scores.shape: ", scores.shape
        # 将上面产生的锚窗+滑窗结果附加到bbox_deltas上面产生最终的兴趣区域
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        print "ProposalLayer proposals: ", proposals
        print "ProposalLayer proposals.shape: ", proposals.shape
        # 将产生的兴趣区域映射回原图
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])
        print "ProposalLayer proposals: ", proposals
        print "ProposalLayer proposals.shape: ", proposals.shape
        # 将超过原图边界的区域，滤除掉
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        print "ProposalLayer keep: ", keep
        print "ProposalLayer keep.shape: ", len(keep)
        print "ProposalLayer proposals: ", proposals
        print "ProposalLayer proposals.shape: ", proposals.shape
        print "ProposalLayer scores: ", scores
        print "ProposalLayer scores.shape: ", scores.shape
        # 将proposal，scores按照scores排序并取出前RPN_PRE_NMS_TOP_N个区域来
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        print "ProposalLayer order: ", order
        print "ProposalLayer order.shape: ", order.shape
        print "ProposalLayer proposals: ", proposals
        print "ProposalLayer proposals.shape: ", proposals.shape
        print "ProposalLayer scores: ", scores
        print "ProposalLayer scores.shape: ", scores.shape
        # 应用nms算法，来进行过滤区域，并最终剩余after_nms_topN个
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        print "ProposalLayer keep: ", keep
        print "ProposalLayer keep.shape: ", len(keep)
        print "ProposalLayer proposals: ", proposals
        print "ProposalLayer proposals.shape: ", proposals.shape
        print "ProposalLayer scores: ", scores
        print "ProposalLayer scores.shape: ", scores.shape
        # 准备结果，送入下一层roi-data层中
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        print "ProposalLayer batch_inds: ", batch_inds
        print "ProposalLayer batch_inds.shape: ", batch_inds.shape
        print "ProposalLayer blob: ", blob
        print "ProposalLayer blob.shape: ", blob.shape
        print "ProposalLayer top[0]: ", top[0]
        print "ProposalLayer top[0].shape: ", top[0].shape
        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
