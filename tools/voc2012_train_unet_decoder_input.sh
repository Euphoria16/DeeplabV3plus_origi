#!/bin/bash
nvidia-smi

# pytorch 04
PYTHON="python"


#dataset config
DATASET="voc2012"

#training setting
BN_MOMENTUM=0.1
GPU="4,5,6,7"

#training settings stage1
LEARNING_RATE_STAGE1=0.007
FREZEE_BN_STAGE1=False
STORE_CHRCKPOINT_NAME_STAGE1="voc2012_aug_no_class_weight"
OUTPUT_STRIDE_STAGE1=16

#training settings stage1
LEARNING_RATE_STAGE2=0.001
FREZEE_BN_STAGE2=True
SAVED_CHECKPOINT_FILE_STAGE2="voc2012_aug_no_class_weight"
STORE_CHRCKPOINT_NAME_STAGE2="voc2012_aug_no_class_weight_2"
OUTPUT_STRIDE_STAGE2=8
PRETRAINED=True
IMAGENET_TRAINED_STAGE2=False
RUN_NAME=voc_Deeplab_unet_input




########################################################################################################################
#  Training
########################################################################################################################
$PYTHON -u train_voc_unet_decoder_input.py --gpu $GPU --freeze_bn $FREZEE_BN_STAGE1 --batch_size_per_gpu 8 \
--bn_momentum $BN_MOMENTUM --lr $LEARNING_RATE_STAGE1 --output_stride $OUTPUT_STRIDE_STAGE1 --run_name $RUN_NAME

#$PYTHON -u train.py --gpu $GPU --store_checkpoint_name $STORE_CHRCKPOINT_NAME_STAGE2 \
#--saved_checkpoint_file $SAVED_CHECKPOINT_FILE_STAGE2 --freeze_bn $FREZEE_BN_STAGE2 --bn_momentum $BN_MOMENTUM \
#--imagenet_pretrained $IMAGENET_TRAINED_STAGE2 --lr $LEARNING_RATE_STAGE2 --output_stride $OUTPUT_STRIDE_STAGE2 \
#--pretrained $PRETRAINED