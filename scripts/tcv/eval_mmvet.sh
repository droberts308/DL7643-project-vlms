#!/bin/bash

python ./modules/ahmad_conditional_visual/eval_vqa.py \
    --model-path ./modules/ahmad_conditional_visual/checkpoints/phi-dora/checkpoint-34137 \
    --question-file ./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/images \
    --answers-file ./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/answers/TCV_phi3.jsonl \
    --temperature 0 \

python ./modules/ahmad_conditional_visual/convert_mmvet_for_eval.py \
    --src ./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/answers/TCV_phi3.jsonl \
    --dst ./modules/ahmad_conditional_visual/data/eval/mmvet/mm-vet/results/TCV_phi3.jsonl