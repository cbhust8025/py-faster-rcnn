#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=1000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# faster-rcnn_end2end zf 10000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_10000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 20000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_20000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 30000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_30000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 40000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_40000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 50000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_50000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 60000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_60000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end zf 70000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_70000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 10000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_10000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 20000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_20000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 30000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_30000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 40000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_40000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 50000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_50000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 60000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_60000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

# faster-rcnn_end2end vgg16 70000 iters
set +x
NET_FINAL="/home/cb/Documents/MyGitHub/py-faster-rcnn/output_server/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel"
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/VGG16/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}


