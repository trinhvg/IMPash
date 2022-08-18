#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5


python main_linear_9sra.py \
 --method PIRL \
 --ckpt ./save/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
 --aug_linear RA \
  --image_size 224\
  --dataset_name  k19 \
  --model_path ./save_9 \
  --tb_path ./tb_9 \
  --n_class 9\
  --keephead head \
 --multiprocessing-distributed --world-size 1 --rank 0 \
 --dist-url 'tcp://127.0.0.1:23457'



#python main_linear_9sra.py \
# --method PatchS \
#--jigsaw_stitch\
# --ckpt ./save/k19_224_PatchS_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --resume ./save_9/k19_224_PatchS_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/best.pth\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead None \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'


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
#  --keephead None \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23457'


#python main_linear_9sra.py \
# --method InsDis \
# --ckpt ./save/k19_224_InsDis_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
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
# --method InsDis \
#  --infer_only \
# --ckpt ./save/k19_224_InsDis_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InsDis_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead None \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'

#python main_infer.py \
# --method InsDis \
#  --infer_only \
# --ckpt ./save/k19_224_InsDis_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InsDis_resnet50_RGB_Jig_False_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead None \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'




#python main_infer.py \
# --method PIRL \
#  --infer_only \
# --ckpt ./save/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead None \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'
#
#python main_infer.py \
# --method PIRL \
#  --infer_only \
# --ckpt ./save/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_PIRL_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_False_JigAug_False_V0_bank_aug_A_linear_0.07_cosine_linear_head_None_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead None \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23455'


#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k19+crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
#python main_linear_finetune.py \
# --aug_linear RA \
# --dataset_name k19+crc \
# --preImageNet \
# --finetune \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_linear_finetune.py \
# --aug_linear RA \
# --dataset_name k16 \
# --preImageNet \
# --finetune \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'



#python main_linear_9sra.py \
# --method InfoMin \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_60.pth\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
#  --aug_linear RA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_60.pth\
#  --aug_linear RA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_40.pth\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#python main_infer_ema.py \
# --method InfoMin \
#  --infer_only \
#  --ckpt ./save//k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_head_head_emahead_False_onk19_RA_0.2_fixedData/ckpt_epoch_60.pth\
#  --aug_linear SRA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224\
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'

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
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name k16\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name crcval\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name crcval\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'
#
#python main_infer.py \
# --infer_only \
# --ckpt_class ./save_9/k19_ImageNet_linear_onk19_224_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
# --dataset_name crcval\
# --image_size 224\
# --model_path ./save_40_9 \
# --tb_path ./save_40_9 \
# --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23459'

#


#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/best.pth\
#  --jigsaw_stitch\
#  --aug_linear RA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/best.pth\
#  --jigsaw_stitch\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23453'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/best.pth\
#  --jigsaw_stitch\
#  --aug_linear RA \
#  --dataset_name k16\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23458'
#
#
#python main_infer_ema.py \
#  --method InfoMin \
#  --infer_only \
#  --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
#  --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/best.pth\
#  --jigsaw_stitch\
#  --aug_linear SRA \
#  --dataset_name crcval\
#  --n_class 9\
#  --keephead head \
#  --image_size 224 \
#  --model_path ./save_40_9\
#  --tb_path ./save_40_9\
#  --multiprocessing-distributed --world-size 1 --rank 0 \
#  --dist-url 'tcp://127.0.0.1:23458'

#
#python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_True_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --jigsaw_stitch\
#    --ema_feat\
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
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_True_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
#  --jigsaw_stitch\
#    --ema_feat\
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
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_True_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
#  --jigsaw_stitch\
#    --ema_feat\
# --dataset_name k16\
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
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_None_emahead_True_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
#  --jigsaw_stitch\
#    --ema_feat\
# --dataset_name k16\
#  --n_class 9\
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

#python main_infer.py \
# --infer_only \
# --preImageNet \
# --ckpt_class ./save/k19+crc_ImageNet_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'

#python main_linear.py \
# --aug_linear RA\
#  --image_size 150\
#  --dataset_name  k19 \
#  --model_path ./save \
#  --tb_path ./tb \
#  --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# --aug_linear RA
# --dataset_name k19
# --preImageNet
# --colorAug
# --multiprocessing-distributed --world-size 1 --rank 0
# --dist-url 'tcp://127.0.0.1:23458'


#python main_infer.py \
# --infer_only \
# --ckpt ./save/k19+crc_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save/k19+crc_224_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine_linear_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16 \
# --colornorm\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23452'


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


#python main_linear_9sra_ema.py \
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
#  --keephead head \
#  --ema_feat\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear RA \
# --dataset_name k16\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear RA \
# --dataset_name k16\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth\
# --aug_linear SRA \
# --dataset_name k16\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'
#
# python main_infer_ema.py \
# --method InfoMin \
# --infer_only \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --ckpt_class ./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_False_JigEMA_True_JigAug_False_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_60.pth\
# --aug_linear SRA \
# --dataset_name k16\
#  --n_class 9\
#  --keephead head \
# --image_size 224\
# --model_path ./save_40_9\
# --tb_path ./save_40_9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23453'

#python main_linear_9sra.py \
# --method InfoMin \
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigEMA_True_V1_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
#  --keephead head \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'


# --method InfoMin
# --ckpt ./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth
# --aug_linear RA
#  --image_size 224
#  --dataset_name  k19
#  --model_path ./save_9
#  --tb_path ./tb_9
#  --n_class 9
# --multiprocessing-distributed --world-size 1 --rank 0

#python main_linear.py \
# --method InfoMin \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --aug_linear RA \
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'


#python main_linear.py \
# --method InfoMin \
# --ckpt ./save/k19_InfoMin_resnet50_RGB_Jig_True_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth\
# --aug_linear RA \
#  --image_size 150\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  --n_class 9\
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'

# python main_linear.py \
# --aug_linear RA\
#  --image_size 224\
#  --dataset_name  k19 \
#  --model_path ./save_9 \
#  --tb_path ./tb_9 \
#  -- n_class 9\
#  --preImageNet \
# --multiprocessing-distributed --world-size 1 --rank 0 \
# --dist-url 'tcp://127.0.0.1:23458'


#
# python main_linear.py \
# --method CMC --ckpt ./save/k19_CMC_resnet50_CMC_Jig_False_bank_aug_C_linear_0.07_cosine/ckpt_epoch_200.pth --aug_linear RA  --dataset_name k19  --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:23459'

#opt.model_name:  CLASS_ce_prostate_hv_effiB0_BS128_lr_0.1_decay_0.0001_seed2_imageS_512_cosine_True_StdPre_ImageNet_and_TecPre_ImageNet_TB0_SB0_BZ64_SupConLoss_SaSa_1WB_weight_0.0001_F2345_out_dim_128_Wscale1_Bscale1_Binit_uniform_RFF_init_gauss01
