### inference with pretrain ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py \
--config mmDiff_config.yml --batch_size 512 \
--model_diff_path checkpoints/checkpoint.pth \ # add your pretrained model
--doc test --exp exp --ni \

### train ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py --train \
--config mmDiff_config.yml --batch_size 512 \
--doc test --exp exp --ni \
