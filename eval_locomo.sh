#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --disable-flash-attn \
# NOTE: chunk-size 256 for Locomo + Qwen
python main.py \
    --memory-cache-suffix "locomo_eval" \
    --eval-only \
    --inference-workers 1 \
    --inference-session-workers 1 \
    --action-top-k 7 \
    --mem-top-k-eval 20 \
    --session-mode full-session \
    --chunk-size 2048 \
    --chunk-overlap 256 \
    --load-checkpoint './checkpoints/locomo_with_designer/locomo-train_epoch_final.pt' \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --model "gpt-4o-mini-ca" \
    --api \
    --api-base "[YOUR_API_BASE]" \
    --api-key "YOUR_API_KEY_1" "YOUR_API_KEY_2" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 100 \
    --outer-epochs 10 \
    --batch-size 4 \
    --encode-batch-size 64 \
    --ppo-epochs 2 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 20 \
    --designer-max-changes 2 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric llm_judge \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --skip-load-snapshot-manager \
    --wandb-run-name eval \
    --save-dir ./checkpoints/locomo_with_designer \
    --disable-flash-attn \
    --out-file ./results/locomo_with_designer.json
