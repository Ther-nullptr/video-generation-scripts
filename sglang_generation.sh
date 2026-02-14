for dimension in subject_consistency human_action object_class
do
    for inference_steps in 5 10 15 20 25 30
    do
        python3 sglang_diffusion_generation_sampled.py --dimensions $dimension --prompts_dir /home/wyj24/project/VBench/prompts/prompts_per_dimension --num_inference_steps $inference_steps  --save_path /home/wyj24/project/video_generation/output --sample_ratio 0.1 --seed 42
    done
done