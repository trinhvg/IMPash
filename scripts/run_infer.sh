#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

python main_infer_ema.py \
 --method PatchSMoco \
  --jigsaw_stitch\
  --infer_only \
 --ckpt ./save/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
  --ckpt_class ./save_9/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
 --aug_linear SRA \
  --image_size 224\
  --dataset_name  k16 \
 --model_path ./save_40_9\
 --tb_path ./save_40_9\
  --n_class 9\
  --keephead None \
 --multiprocessing-distributed --world-size 1 --rank 0 \
 --dist-url 'tcp://127.0.0.1:23452'


python main_infer_ema.py \
 --method PatchSMoco \
  --jigsaw_stitch\
  --infer_only \
 --ckpt ./save/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
  --ckpt_class ./save_9/k19_224_PatchSMoco_resnet50_RGB_Jig_False_JigStitch_True_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
 --aug_linear SRA \
  --image_size 224\
  --dataset_name  k16 \
 --model_path ./save_40_9\
 --tb_path ./save_40_9\
  --n_class 9\
  --keephead head \
 --multiprocessing-distributed --world-size 1 --rank 0 \
 --dist-url 'tcp://127.0.0.1:23452'



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
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
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
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#

# 2048-9
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead None \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23458'





#python main_linear_9sra.py \
# --method InsDis \
# --ckpt ./save/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
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
# --ckpt_class ./save_9/k19_ImageNet_linear_head_None_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name k16\
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
#
# python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_head_None_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_head_None_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name k16\
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
# --preImageNet \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
#
# python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_head_None_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
# --preImageNet \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'



#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#python main_infer.py \
# --infer_only \
# --method CMC \
# --ckpt ./save/k19_CMC_resnet50_CMC_Jig_False_bank_aug_C_linear_0.07_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19_CMC_resnet50_CMC_Jig_False_bank_aug_C_linear_0.07_cosine_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crc \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
#python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k16_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16val \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
#python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k19val \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#




#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_linear_onk19_150_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
#   --image_size 224\
#     --model_path ./save_40 \
#  --tb_path ./tb_infer \
# --preImageNet \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --image_size 224\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k16_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 224\
# --model_path ./save_40\
# --tb_path ./tb_infer\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'


#python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k19_150_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19_150_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_150_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 150\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k16_colorAug_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --image_size 224\
# --model_path ./save_40_30val \
# --tb_path ./tb_infer_30val \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
##
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/crc_ImageNet_colorAug_linear_oncrc_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#

#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23452'
#

# python main_infer.py \
# --infer_only \
#  --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --image_size 224\
#   --n_class 9\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23455'


#python main_infer_ema.py \
# --infer_only \
#  --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --image_size 224\
#   --n_class 9\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
#   --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'


#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23454'



#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
#  --jigsaw_stitch\
# --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'







# python main_infer.py \
# --infer_only \
#  --ckpt ./save/k19_150_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_150_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_150_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --image_size 150\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#



#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

# python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_150_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
# --image_size 150\
# --model_path ./save_40_9 \
# --tb_path ./tb_infer_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 224\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
# python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 224\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --preImageNet \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
#python main_infer_finetune.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_finetuneV2_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 224\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
# python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16val\
# --image_size 224\
# --model_path ./save_40 \
# --tb_path ./tb_infer \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
# python main_infer.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_linear_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --preImageNet \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
#python main_infer_finetune.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_finetuneV2_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#
#
#python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k19+crc_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19+crc_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23454'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#python main_infer_finetune.py \
# --infer_only \
# --ckpt_class ./save/crc_ImageNet_finetuneV2_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer_finetune.py \
# --infer_only \
# --ckpt_class ./save/k19+crc_ImageNet_finetuneV2_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
#  --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#python main_infer_finetune.py \
# --infer_only \
# --ckpt_class ./save/k19_ImageNet_finetuneV2_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k19val \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'



#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k19+crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer.py \
# --infer_only \
# --method InfoMin \
# --ckpt ./save/k16_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/crc_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k19_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --colornorm\
# --dataset_name crc \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k16_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --colornorm\
# --dataset_name crc \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k16_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --colornorm\
# --dataset_name k19 \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --colornorm\
# --dataset_name k19 \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'
#
#
## --colornorm\




#--infer_only
#--method InfoMin
#--preImageNet
#--ckpt_class ./save/k19_ImageNet_finetune_RA_0.2/ckpt_epoch_60.pth
#--aug_linear RA
#--dataset_name k16
#--multiprocessing-distributed --world-size 1 --rank 0
#--dist-url 'tcp://127.0.0.1:23459'

#
# python main_linear.py \
# --method CMC --ckpt ./save/k19_CMC_resnet50_CMC_Jig_False_bank_aug_C_linear_0.07_cosine/ckpt_epoch_200.pth --aug_linear RA  --dataset_name k19  --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:23459'

# --infer_only
# --method InfoMin
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth
# --ckpt_class ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_60.pth
# --aug_linear RA
# --dataset_name k16
# --multiprocessing-distributed --world-size 1 --rank 0


#eval epoch 0, total time 64.41
#acc 0.6555428571428571
#acc [0.6555428571428571]
#conf_mat [array([[491,  13,   0,  46,  69,   5,   1],
#       [  7, 584,   1,  12,  19,   2,   0],
#       [ 24,  46, 232, 101, 222,   0,   0],
#       [ 29, 183,   6, 361,  34,  11,   1],
#       [  8,   2,   0,  13, 602,   0,   0],
#       [  6,  34,   1, 115,   1, 457,  11],
#       [  7, 264,   0, 210,   1,   2, 141]])]
