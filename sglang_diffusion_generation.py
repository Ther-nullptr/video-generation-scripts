import argparse
import os
import torch
import random
import numpy as np
from tqdm import tqdm

try:
    from sglang.multimodal_gen import DiffGenerator
    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("Warning: 'sglang' library not found. Using dummy noise generation.")

# Global model variable to avoid reloading per sample
generator = None

def parse_args():
    parser = argparse.ArgumentParser(description="VBench Video Generation Script with SGLang")
    
    parser.add_argument(
        "--dimensions",
        nargs='+',
        required=True,
        help="List of evaluation dimensions to generate videos for (e.g., object_class overall_consistency)",
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Root directory to save generated videos",
    )
    
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="./prompts/prompts_per_dimension",
        help="Directory containing prompt files (default: ./prompts/prompts_per_dimension)",
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
        default=50,
        help="Number of denoising steps (default: 50)",
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
        help="Video height (default: 768)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 1360)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Path to the pre-trained model",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
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

def sample_func(prompt, index, base_seed, num_inference_steps=25, height=768, width=1360, output_path=None):
    """
    Interface: Generate a video from a prompt using SGLang.
    """
    global generator
    # Load model is handled outside in this adaptation for better control
    
    if generator == "dummy" or not HAS_SGLANG:
        # print(f"[Dummy] Generating noise video for: '{prompt}'")
        # In dummy mode we just return None and let it pass for now or handle appropriately
        return None

    current_seed = base_seed + index
    
    # print(f"[Generating] Prompt: '{prompt}' | Seed: {current_seed} | Steps: {num_inference_steps} | Res: {width}x{height}")
    
    # SGLang internal handling
    # Note: SGLang's generator.generate might save output internally if output_path is provided.
    # We use the provided output_path to ensure it saves to the correct location directly if possible.
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=current_seed,
            return_frames=False,
            output_path=output_path, # SGLang usually takes a dir or filename base
            save_output=True
        )
    )
    
    return video

def main():
    args = parse_args()
    
    # Append config to save_path
    model_name = os.path.basename(args.model_path).replace("/", "_")
    args.save_path = os.path.join(args.save_path, model_name)
    args.save_path = f"{args.save_path}_steps{args.num_inference_steps}_fps{args.fps}_samples{args.num_samples}_res{args.width}x{args.height}"
    
    if args.enable_cache_dit:
        args.save_path += "_cache_dit"
        if args.cache_dit_rdt is not None:
            args.save_path += f"_rdt{args.cache_dit_rdt}"
        if args.cache_dit_scm_preset:
            args.save_path += f"_scm{args.cache_dit_scm_preset}"
        if args.cache_dit_fn is not None:
            args.save_path += f"_fn{args.cache_dit_fn}"
        if args.cache_dit_bn is not None:
            args.save_path += f"_bn{args.cache_dit_bn}"
        if args.cache_dit_warmup is not None:
            args.save_path += f"_warmup{args.cache_dit_warmup}"
        if args.cache_dit_mc is not None:
            args.save_path += f"_mc{args.cache_dit_mc}"
        if args.cache_dit_taylorseer:
            args.save_path += "_taylorseer"
        if args.cache_dit_ts_order is not None:
            args.save_path += f"_tsorder{args.cache_dit_ts_order}"
            
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

    load_model(args.model_path, args.num_gpus)

    # Collect all tasks
    tasks = []
    for dimension in args.dimensions:
        prompt_file = os.path.join(args.prompts_dir, f"{dimension}.txt")
        if not os.path.exists(prompt_file):
            print(f"Warning: Prompt file not found at {prompt_file}, skipping...")
            continue
            
        with open(prompt_file, 'r') as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        
        dim_save_path = os.path.join(args.save_path, dimension)
        os.makedirs(dim_save_path, exist_ok=True)

        for prompt in prompt_list:
            for index in range(args.num_samples):
                tasks.append((dimension, prompt, index, dim_save_path))

    # Run tasks with progress bar
    for dimension, prompt, index, dim_save_path in tqdm(tasks, desc="Generating Videos"):
        # We sanitize prompt for filename
        safe_prompt = "".join([c if c.isalnum() or c in (' ', '_', '-') else '' for c in prompt]).replace(' ', '_')[:100]
        filename = f"{safe_prompt}-{index}.mp4"
        file_path = os.path.join(dim_save_path, filename)
        
        if os.path.exists(file_path):
            continue

        sample_func(
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