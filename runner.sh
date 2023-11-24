### train ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py \
--config mmDiff_config.yml --batch_size 512 \
--model_diff_path checkpoints/checkpoint.pth \
--doc test --exp exp --ni \
