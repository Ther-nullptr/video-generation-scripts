for dimension in appearance_style color human_action multiple_objects object_class scene spatial_relationship subject_consistency temporal_flickering temporal_style; do
    python3 sglang_diffusion_generation.py --dimensions $dimension --save_path output/Wan2.1-T2V-1.3B-Diffusers_$dimension --prompts_dir /home/wyj24/project/VBench/prompts/prompts_per_dimension
done