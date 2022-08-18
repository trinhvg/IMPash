#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3


python main_contrast_Jig.py \
  --method PatchSMoco \
    --jigsaw_stitch\
  --cosine \
   --image_size 224\
   --dataset_name  k19 \
   --model_path ./save_dump \
   --tb_path ./tb_dump \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dist-url 'tcp://127.0.0.1:23458'

##python main_contrast_Jig.py \
#  --method PatchSMoco
#    --jigsaw_stitch
#  --cosine
#   --image_size 224
#   --dataset_name  k19
#   --model_path ./save_dump
#   --tb_path ./tb_dump
#  --multiprocessing-distributed --world-size 1 --rank 0
#  --dist-url 'tcp://127.0.0.1:23458'




# 2048-9
#python main_infer_ema.py \
# --method PatchSMem \
#  --jigsaw_stitch\
#  --infer_only \
# --ckpt ./save/k19_224_PatchSMem_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PatchSMem_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --image_size 224\
#  --dataset_name  k16 \
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
#  --n_class 9\
#  --keephead None \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
# --method PatchSMem \
#  --jigsaw_stitch\
#  --infer_only \
# --ckpt ./save/k19_224_PatchSMem_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PatchSMem_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --image_size 224\
#  --dataset_name  k16 \
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

#python main_infer_ema.py \
# --method PIRL \
# --infer_only \
# --ckpt ./save/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --keephead head \
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --keephead head \
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'





#python main_linear_9sra.py \
# --method PatchSMem \
#--jigsaw_stitch\
# --ckpt ./save/k19_224_PatchSMem_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23452'



#python main_infer_ema.py \
# --method PatchSMoco \
#  --jigsaw_stitch\
#  --infer_only \
# --ckpt ./save/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --image_size 224\
#  --dataset_name  crcval \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead None \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23452'
#
#
#python main_infer_ema.py \
# --method PatchSMoco \
#  --jigsaw_stitch\
#  --infer_only \
# --ckpt ./save/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --image_size 224\
#  --dataset_name  crcval \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23452'








#python main_contrast_Jig.py \
#  --method PatchSMoco \
#    --jigsaw_stitch\
#  --cosine \
#   --image_size 224\
#   --dataset_name  k19 \
#   --model_path ./save \
#   --tb_path ./tb \
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23458'

#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --keephead head \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --keephead head \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --keephead head \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
# --keephead head \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --keephead head \
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
# --keephead head \
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'

#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt         ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_True_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --jigsaw_stitch\
#      --ema_feat\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_True_onk19_RA_0.2/ckpt_epoch_60.pth\
#  --jigsaw_stitch\
#  --ema_feat\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_True_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --jigsaw_stitch\
#  --ema_feat\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_True_onk19_RA_0.2/ckpt_epoch_60.pth\
#  --jigsaw_stitch\
#  --ema_feat\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'


#python main_linear_9sra_ema_nohead.py \
# --method InfoMin \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --jigsaw_ema\
#  --jigsaw_stitch\
# --jig_version V0\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead None \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'



#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k19+crc \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23454'


#python main_linear.py \
# --aug_linear RA \
# --dataset_name k16 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23451'


#python main_contrast_Jig.py \
#   --method InfoMin \
#   --jigsaw_ema\
#   --jigsaw_stitch\
#   --jig_version V0\
#   --cosine \
#   --image_size 224\
#   --dataset_name  k19 \
#   --model_path ./save \
#   --tb_path ./tb \
#   --multiprocessing-distributed --world-size 1 --rank 0\
#   --dist-url 'tcp://127.0.0.1:23453'

#python main_contrast_2.py \
#   --method InfoMin \
#   --jigsaw_ema\
#   --jig_version V0\
#   --cosine \
#   --image_size 224\
#   --dataset_name  k19 \
#   --model_path ./save \
#   --tb_path ./tb \
#   --multiprocessing-distributed --world-size 1 --rank 0\
#   --dist-url 'tcp://127.0.0.1:23452'


#python main_contrast_2.py \
#   --method InfoMin \
#   --jigsaw_ema\
#   --jig_version V1\
#   --cosine \
#   --image_size 224\
#   --dataset_name  k19 \
#   --model_path ./save \
#   --tb_path ./tb \
#   --multiprocessing-distributed --world-size 1 --rank 0\
#   --dist-url 'tcp://127.0.0.1:23452'


#--image_size 150 --method InfoMin     --cosine     --dataset_name  k19    --model_path ./save_dump --tb_path ./tb_dump   --multiprocessing-distributed --world-size 1 --rank 0

#python main_contrast.py \
#  --method InsDis \
#  --cosine \
#   --image_size 224\
#   --dataset_name  k19 \
#   --model_path ./save \
#   --tb_path ./tb \
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'

#python main_contrast.py \
#   --model_path ./save_dump
#   --method InfoMin
#   --cosine
#   --dataset_name k19
#   --multiprocessing-distributed --world-size 1 --rank 0
#   --dist-url 'tcp://127.0.0.1:23456'

