#!/bin/bash
# Commands for the experiments reported in paper/eccv2022submissionCR.tex.
# Reconstructed from the original run_cons/run_linear/run_infer scripts; run from the repo root.
# Usage: bash scripts/run_paper.sh {pretrain|linear|infer|umap|export} [METHOD]
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
DIST="--multiprocessing-distributed --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:23453"

# Table 1 / Table 2 encoders (pretrained on K19 for 200 epochs)
#   IMPaSh                       : --method InfoMin --jigsaw_stitch --jigsaw_ema --jig_version V0
#   InfoMin + PatchShuffling (O) : --method InfoMin --jigsaw_stitch --jig_version V0
#   InfoMin + PIRL + ME          : --method InfoMin --jigsaw_ema --jig_version V0
#   InfoMin (= InfoMin + PIRL)   : --method InfoMin
#   MoCoV2 / PIRL / InsDis       : --method MoCov2 | PIRL | InsDis
SSL_FLAGS=${SSL_FLAGS:---method InfoMin --jigsaw_stitch --jigsaw_ema --jig_version V0}

# encoder checkpoint + linear classifier used for IMPaSh in the paper
CKPT=${CKPT:-./save/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine/ckpt_epoch_200.pth}
CKPT_CLASS=${CKPT_CLASS:-./save_9/k19_224_InfoMin_resnet50_RGB_Jig_True_JigStitch_True_JigEMA_True_V0_moco_aug_D_mlp_0.15_cosine_linear_head_emahead_False_onk19_RA_0.2/ckpt_epoch_40.pth}
KEEPHEAD=${KEEPHEAD:-head}   # 'head' keeps projector p1 (Sec. 2.4); baselines use None

case "$1" in
  pretrain)
    python main_contrast_impash.py $SSL_FLAGS \
      --cosine --image_size 224 --dataset_name k19 \
      --model_path ./save --tb_path ./tb $DIST ;;
  linear)
    # ImageNet baselines: drop --ckpt and add --preImageNet (train on k19 = lower bound, k16 = upper bound)
    python main_linear_impash.py $SSL_FLAGS --ckpt $CKPT \
      --aug_linear RA --image_size 224 --dataset_name k19 --n_class 9 \
      --keephead $KEEPHEAD --model_path ./save_9 --tb_path ./tb_9 $DIST ;;
  infer)
    # test on K16 (target domain); add --colornorm for the stain-normalisation baselines
    python main_infer_impash.py $SSL_FLAGS --infer_only --ckpt $CKPT --ckpt_class $CKPT_CLASS \
      --aug_linear SRA --image_size 224 --dataset_name k16 --n_class 9 \
      --keephead $KEEPHEAD --model_path ./save_40_9 --tb_path ./save_40_9 $DIST ;;
  umap)
    # Fig. 3 (UMAP) and Table 3 (silhouette scores) -> viz_final_eccv/${FIG_NAME}*
    # FIG_NAME defaults to 9 so the paper panels (0, 1, 2, 5) are not overwritten
    python main_umap_impash.py $SSL_FLAGS --infer_only --ckpt $CKPT --ckpt_class $CKPT_CLASS \
      --aug_linear SRA --image_size 224 --dataset_name k16 --n_class 9 --fig_name ${FIG_NAME:-9} \
      --keephead $KEEPHEAD --model_path ./save_umap --tb_path ./save_umap $DIST ;;
  export)
    # collect Acc / Re / Pre / F1 from ./save_40_9_infer_40 into a CSV
    python export_acc/export_acc_9.py ;;
  *)
    echo "usage: $0 {pretrain|linear|infer|umap|export}"; exit 1 ;;
esac
