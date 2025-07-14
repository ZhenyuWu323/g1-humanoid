export CUDA_VISIBLE_DEVICES=1

python scripts/rsl_rl/train.py \
    --task Template-G1-WholeBody-Plate-Locomanipulation \
    --num_envs 10 \
    --headless

python scripts/rsl_rl/train.py \
    --task Template-G1-WholeBody-Plate-Locomanipulation \
    --num_envs 4096 \
    --headless
