import argparse
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import json

from sglang.multimodal_gen import DiffGenerator
HAS_SGLANG = True


# Global model variable to avoid reloading per sample
generator = None

def parse_args():
    parser = argparse.ArgumentParser(description="VBench I2V Video Generation Script with SGLang")
    
    parser.add_argument(
        "--dimensions",
        nargs='+',
        required=True,
        help="List of evaluation dimensions to generate videos for (e.g., i2v_subject i2v_background camera_motion)",
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Root directory to save generated videos",
    )
    
    parser.add_argument(
        "--i2v_info_json",
        type=str,
        default="./vbench2_beta_i2v/vbench2_i2v_full_info.json",
        help="Path to VBench-I2V full info JSON file",
    )
    
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./vbench2_beta_i2v/data/crop",
        help="Root folder containing cropped images",
    )
    
    parser.add_argument(
        "--resolution",
        type=str,
        default="1-1",
        choices=["1-1", "8-5", "7-4", "16-9"],
        help="Image resolution ratio (default: 1-1)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of video samples per prompt (default: 5)",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="FPS for saved videos (default: 16)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.1-I2V-14B-Diffusers",
        help="Path to the pre-trained I2V model",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )

    # Sampling arguments
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of prompts to sample from each dimension (0.0-1.0, default: 1.0 = all prompts)",
    )
    
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Random seed for sampling prompts (default: use --seed value)",
    )

    # Cache-DiT arguments
    parser.add_argument("--enable-cache-dit", action="store_true", help="Enable Cache-DiT acceleration")
    parser.add_argument("--cache-dit-rdt", type=float, help="Cache-DiT Residual Difference Threshold (RDT)")
    parser.add_argument("--cache-dit-scm-preset", type=str, help="Cache-DiT SCM Preset (none, slow, medium, fast, ultra)")
    parser.add_argument("--cache-dit-fn", type=int, help="Cache-DiT First N blocks to always compute")
    parser.add_argument("--cache-dit-bn", type=int, help="Cache-DiT Last N blocks to always compute")
    parser.add_argument("--cache-dit-warmup", type=int, help="Cache-DiT Warmup steps")
    parser.add_argument("--cache-dit-mc", type=int, help="Cache-DiT Max continuous cached steps")
    parser.add_argument("--cache-dit-taylorseer", action="store_true", help="Enable TaylorSeer for Cache-DiT")
    parser.add_argument("--cache-dit-ts-order", type=int, help="Cache-DiT TaylorSeer order")

    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_model(model_path, num_gpus):
    """
    Initialize and load your SGLang generator here.
    """
    global generator
    if HAS_SGLANG:
        print(f"Loading SGLang DiffGenerator from {model_path} with {num_gpus} GPUs...")
        generator = DiffGenerator.from_pretrained(
            model_path=model_path,
            num_gpus=num_gpus,
        )
        print("Model loaded successfully.")
    else:
        generator = "dummy"

def load_i2v_data(i2v_info_json, dimensions, resolution):
    """
    Load I2V data from the JSON file for specified dimensions.
    
    Returns:
        List of tuples: (image_name, prompt, dimension)
    """
    with open(i2v_info_json, 'r') as f:
        data = json.load(f)
    
    inputs = []
    for item in data:
        item_dimensions = item.get("dimension", [])
        # Check if any of the requested dimensions are in this item
        for dim in dimensions:
            if dim in item_dimensions:
                image_name = item.get("image_name", "")
                prompt = item.get("prompt_en", "")
                if image_name and prompt:
                    inputs.append((image_name, prompt, dim))
                break  # Only add once per item
    
    return inputs

def sample_prompts(inputs, sample_ratio, sample_seed):
    """
    Sample a subset of inputs based on the given ratio and seed.
    
    Args:
        inputs: List of (image_name, prompt, dimension) tuples
        sample_ratio: Float between 0.0 and 1.0, representing the percentage to sample
        sample_seed: Random seed for sampling
    
    Returns:
        List of sampled inputs
    """
    if sample_ratio >= 1.0:
        return inputs
    
    if sample_ratio <= 0.0:
        return []
    
    # Create a separate random generator for sampling with the given seed
    rng = random.Random(sample_seed)
    
    # Calculate number of prompts to sample
    total_prompts = len(inputs)
    num_to_sample = max(1, int(total_prompts * sample_ratio))
    num_to_sample = min(num_to_sample, total_prompts)  # Ensure we don't exceed total
    
    # Sample without replacement
    sampled = rng.sample(inputs, num_to_sample)
    
    return sampled

def sample_func(image_path, prompt, index, base_seed, num_inference_steps=50, height=480, width=832, output_path=None):
    """
    Interface: Generate a video from an image and prompt using SGLang.
    
    Args:
        image_path: Path to the input image
        prompt: Text prompt describing the desired video
        index: Sample index for seed calculation
        base_seed: Base random seed
        num_inference_steps: Number of denoising steps
        height: Video height
        width: Video width
        output_path: Path to save the output video
    
    Returns:
        Generated video
    """
    global generator
    
    if generator == "dummy" or not HAS_SGLANG:
        return None

    current_seed = base_seed + index
    
    # SGLang I2V generation
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt,
            image_path=image_path,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=current_seed,
            return_frames=False,
            output_path=output_path,
            save_output=True
        )
    )
    
    return video

def main():
    args = parse_args()
    
    # Build output directory with explicit run config information
    model_name = os.path.basename(args.model_path).replace("/", "_")
    run_tag = (
        f"steps{args.num_inference_steps}"
        f"_fps{args.fps}"
        f"_samples{args.num_samples}"
        f"_res{args.width}x{args.height}"
        f"_ratio{args.resolution}"
        f"_seed{args.seed}"
    )
    
    # Add sampling info to save_path if not sampling all
    if args.sample_ratio < 1.0:
        effective_sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
        run_tag += f"_sample{args.sample_ratio:.2f}_sampleseed{effective_sample_seed}"
    
    if args.enable_cache_dit:
        run_tag += "_cache_dit"
        if args.cache_dit_rdt is not None:
            run_tag += f"_rdt{args.cache_dit_rdt}"
        if args.cache_dit_scm_preset:
            run_tag += f"_scm{args.cache_dit_scm_preset}"
        if args.cache_dit_fn is not None:
            run_tag += f"_fn{args.cache_dit_fn}"
        if args.cache_dit_bn is not None:
            run_tag += f"_bn{args.cache_dit_bn}"
        if args.cache_dit_warmup is not None:
            run_tag += f"_warmup{args.cache_dit_warmup}"
        if args.cache_dit_mc is not None:
            run_tag += f"_mc{args.cache_dit_mc}"
        if args.cache_dit_taylorseer:
            run_tag += "_taylorseer"
        if args.cache_dit_ts_order is not None:
            run_tag += f"_tsorder{args.cache_dit_ts_order}"

    args.save_path = os.path.join(args.save_path, model_name, run_tag)
            
    print(f"Output directory updated to: {args.save_path}")
    
    # Set Cache-DiT environment variables
    if args.enable_cache_dit:
        os.environ["SGLANG_CACHE_DIT_ENABLED"] = "true"
    else:
        os.environ["SGLANG_CACHE_DIT_ENABLED"] = "false"

    def set_env_or_pop(key, value):
        if value is not None:
            os.environ[key] = str(value)
        else:
            os.environ.pop(key, None)

    set_env_or_pop("SGLANG_CACHE_DIT_RDT", args.cache_dit_rdt)
    set_env_or_pop("SGLANG_CACHE_DIT_SCM_PRESET", args.cache_dit_scm_preset)
    set_env_or_pop("SGLANG_CACHE_DIT_FN", args.cache_dit_fn)
    set_env_or_pop("SGLANG_CACHE_DIT_BN", args.cache_dit_bn)
    set_env_or_pop("SGLANG_CACHE_DIT_WARMUP", args.cache_dit_warmup)
    set_env_or_pop("SGLANG_CACHE_DIT_MC", args.cache_dit_mc)

    if args.cache_dit_taylorseer:
        os.environ["SGLANG_CACHE_DIT_TAYLORSEER"] = "true"
    else:
        os.environ.pop("SGLANG_CACHE_DIT_TAYLORSEER", None)

    set_env_or_pop("SGLANG_CACHE_DIT_TS_ORDER", args.cache_dit_ts_order)

    if args.seed is not None:
        seed_everything(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load model
    load_model(args.model_path, args.num_gpus)

    # Determine effective sample seed
    effective_sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    print(f"Sampling configuration: ratio={args.sample_ratio}, seed={effective_sample_seed}")

    # Collect all tasks
    tasks = []
    for dimension in args.dimensions:
        # Load data for this dimension
        inputs = load_i2v_data(args.i2v_info_json, [dimension], args.resolution)
        
        if not inputs:
            print(f"Warning: No data found for dimension '{dimension}', skipping...")
            continue
        
        # Sample inputs for this dimension independently
        original_count = len(inputs)
        sampled_inputs = sample_prompts(inputs, args.sample_ratio, effective_sample_seed)
        sampled_count = len(sampled_inputs)
        
        print(f"Dimension '{dimension}': {sampled_count}/{original_count} prompts sampled ({args.sample_ratio*100:.1f}%)")
        
        dim_save_path = os.path.join(args.save_path, dimension)
        os.makedirs(dim_save_path, exist_ok=True)

        # Create tasks
        resolution_folder = os.path.join(args.image_folder, args.resolution)
        for image_name, prompt, dim in sampled_inputs:
            image_path = os.path.join(resolution_folder, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping...")
                continue
            for index in range(args.num_samples):
                tasks.append((dimension, image_path, prompt, index, dim_save_path))

    print(f"\nTotal tasks to process: {len(tasks)}")

    # Run tasks with progress bar
    for dimension, image_path, prompt, index, dim_save_path in tqdm(tasks, desc="Generating Videos"):
        # Sanitize prompt for filename
        safe_prompt = "".join([c if c.isalnum() or c in (' ', '_', '-') else '' for c in prompt]).replace(' ', '_')[:100]
        filename = f"{safe_prompt}-{index}.mp4"
        file_path = os.path.join(dim_save_path, filename)
        
        if os.path.exists(file_path):
            continue

        sample_func(
            image_path=image_path,
            prompt=prompt, 
            index=index, 
            base_seed=args.seed, 
            num_inference_steps=args.num_inference_steps, 
            height=args.height, 
            width=args.width,
            output_path=file_path
        )

    print("Generation complete.")

    # Explicitly clean up generator to avoid __del__ issues during interpreter shutdown
    global generator
    if generator is not None and generator != "dummy":
        del generator
        generator = None

if __name__ == "__main__":
    main()
