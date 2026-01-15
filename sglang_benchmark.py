import argparse
import os
import torch
import random
import numpy as np
import time
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
    parser = argparse.ArgumentParser(description="VBench Video Generation Benchmark with SGLang")
    
    # Keeping original args for compatibility and model loading
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

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Benchmark specific args
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=2,
        help="Number of warmup runs before benchmarking",
    )
    
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=5,
        help="Number of runs to average for benchmarking",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a cute cat playing with a red ball on a green grass field, 4k, high quality.",
        help="Prompt to use for benchmarking",
    )
    
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Whether to save the output video during benchmark (default: False to measure pure generation)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_output",
        help="Directory to save output if --save_output is set",
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

def main():
    args = parse_args()
    
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

    load_model(args.model_path, args.num_gpus)
    
    if generator == "dummy" or not HAS_SGLANG:
        print("SGLang not found or dummy mode. Exiting.")
        return

    # Prepare output dir if saving
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)

    # Warmup
    print(f"\n[Warmup] Running {args.warmup_runs} warmup generations...")
    for i in range(args.warmup_runs):
        generator.generate(
            sampling_params_kwargs=dict(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                seed=args.seed + i, # Vary seed slightly
                return_frames=False,
                output_path=os.path.join(args.output_dir, f"warmup_{i}.mp4") if args.save_output else None,
                save_output=args.save_output
            )
        )
    print("Warmup complete.")

    # Benchmark
    print(f"\n[Benchmark] Running {args.benchmark_runs} generations for measurement...")
    latencies = []
    diffusion_latencies = []
    
    class DiffusionTimer:
        def __init__(self):
            self.step_times = []
            
        def __call__(self, *args, **kwargs):
            self.step_times.append(time.time())
            # Return the last arg if it's a dict (callback_kwargs), else empty dict
            if args and isinstance(args[-1], dict):
                return args[-1]
            return {}

    for i in tqdm(range(args.benchmark_runs), desc="Benchmarking"):
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        timer = DiffusionTimer()
        
        # Prepare callback kwargs
        # We try to pass it as both 'callback' (old) and 'callback_on_step_end' (new)
        # hoping one works and the other is ignored or both point to same.
        # However, passing both might cause issues if the underlying pipe checks for conflicts.
        # We'll assume 'callback_on_step_end' is preferred for newer diffusers.
        # But to be safe and simple, let's try 'callback_on_step_end' first.
        
        # Note: SGLang DiffGenerator.generate takes sampling_params_kwargs which are passed to the pipeline.
        
        start_time = time.time()
        
        try:
            generator.generate(
                sampling_params_kwargs=dict(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    seed=args.seed + 100 + i,
                    return_frames=False,
                    output_path=os.path.join(args.output_dir, f"bench_{i}.mp4") if args.save_output else None,
                    save_output=args.save_output,
                    callback_on_step_end=timer,
                    # callback_steps=1 # Ensure callback is called every step
                )
            )
        except Exception as e:
            # Fallback if callback_on_step_end fails (e.g. old diffusers or not supported)
            # print(f"Callback failed: {e}, trying without...")
            generator.generate(
                sampling_params_kwargs=dict(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    seed=args.seed + 100 + i,
                    return_frames=False,
                    output_path=os.path.join(args.output_dir, f"bench_{i}.mp4") if args.save_output else None,
                    save_output=args.save_output
                )
            )
        
        # Synchronize after timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        latencies.append(end_time - start_time)
        
        # Calculate diffusion time estimation
        if len(timer.step_times) > 1:
            # We have timestamps for steps. 
            # If we captured N steps:
            # Time from Step 1 end to Step N end = (N-1) intervals.
            # Avg interval = (t_last - t_first) / (count - 1)
            # Total est = Avg interval * count
            
            # NOTE: timer.step_times records time *after* each step usually.
            count = len(timer.step_times)
            if count > 1:
                duration_measured = timer.step_times[-1] - timer.step_times[0]
                avg_step = duration_measured / (count - 1)
                est_diffusion = avg_step * count # Estimate including the first step
                diffusion_latencies.append(est_diffusion)
            else:
                diffusion_latencies.append(0)
        else:
            diffusion_latencies.append(0)

    # Statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0
    
    avg_diff_latency = np.mean(diffusion_latencies) if diffusion_latencies and any(diffusion_latencies) else 0
    
    print("\n" + "="*40)
    print(f"Benchmark Results ({args.model_path})")
    print("="*40)
    print(f"Configuration:")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Cache-DiT: {args.enable_cache_dit}")
    print("-"*40)
    print(f"Performance:")
    print(f"  Total Latency:     {avg_latency:.4f} s  (std: {std_latency:.4f} s)")
    if avg_diff_latency > 0:
        print(f"  Diffusion Latency: {avg_diff_latency:.4f} s  (approx)")
        print(f"  Other Latency:     {avg_latency - avg_diff_latency:.4f} s  (VAE, TextEnc, etc.)")
    else:
        print(f"  Diffusion Latency: N/A (Callback not supported)")
    print(f"  Total Throughput:  {throughput:.4f} videos/s")
    print("="*40)

    # Cleanup
    global generator
    del generator
    generator = None

if __name__ == "__main__":
    main()
