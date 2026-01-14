import argparse
import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# [User Configuration] Import your model libraries here
# -----------------------------------------------------------------------------
try:
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("Warning: 'diffusers' library not found. Using dummy noise generation.")
    print("To use the real model example, please run: pip install diffusers accelerate")

# Global model variable to avoid reloading per sample
pipe = None

def parse_args():
    parser = argparse.ArgumentParser(description="VBench Video Generation Script")
    
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
        default=25,
        help="Number of denoising steps (default: 25)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="FPS for saved videos (default: 8)",
    )

    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable Flash Attention 2 backend (requires flash-attn installed)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Video height (default: 768)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="Video width (default: 1360)",
    )

    return parser.parse_args()

def process_video_data(video_data):
    # 1. 如果是 CogVideoXPipelineOutput 对象，取出 frames
    if hasattr(video_data, "frames"):
        video_data = video_data.frames[0] # 取第一条视频

    # 2. 如果是 PyTorch Tensor (C, F, H, W) 或 (F, C, H, W) 等
    if isinstance(video_data, torch.Tensor):
        # 移到 CPU
        video_data = video_data.detach().cpu()
        # 处理维度: 确保是 (Frames, Height, Width, Channels)
        if video_data.ndim == 5: # (Batch, C, F, H, W) -> 取 Batch 0
             video_data = video_data[0]
        if video_data.shape[0] < video_data.shape[1]: # 猜测是 (C, F, H, W) -> 转 (F, H, W, C)
            video_data = video_data.permute(1, 2, 3, 0)
        
        # 反归一化 (如果数据在 [-1, 1] 或 [0, 1])
        if video_data.dtype != torch.uint8:
            video_data = (video_data * 255).clamp(0, 255).to(torch.uint8)
        
        # 转为 Numpy
        video_data = video_data.numpy()

    # 3. 如果是 List[Tensor]，转为 List[Numpy]
    if isinstance(video_data, list) and isinstance(video_data[0], torch.Tensor):
        video_data = [frame.cpu().numpy().astype(np.uint8) for frame in video_data]

    return video_data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_model(flash_attn=False):
    """
    Initialize and load your model here.
    """
    global pipe
    if HAS_DIFFUSERS:
        print(f"Loading ModelScope Text-to-Video model (FlashAttention: {flash_attn})...")
        
        # Flash Attention 2 configuration
        kwargs = {
            "torch_dtype": torch.float16,
        }
        if flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        # Using standard ModelScope T2V from Hugging Face
        pipe = DiffusionPipeline.from_pretrained("/home/jl-chen22/CogVideo/ckpt/CogVideoX1.5-5B", **kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Memory optimizations
        pipe.enable_model_cpu_offload() # Saves GPU memory
        # pipe.enable_vae_tiling()       # Crucial for high-res video decoding
        # pipe.enable_vae_slicing()      # Reduces VAE memory usage
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        
        print("Model loaded successfully with VAE tiling/slicing enabled.")
    else:
        pipe = "dummy"

def sample_func(prompt, index, base_seed, num_inference_steps=25, flash_attn=False, height=480, width=720):
    """
    Your Interface: Generate a video tensor from a prompt.
    """
    global pipe
    if pipe is None:
        load_model(flash_attn=flash_attn)
    
    # 1. Dummy Implementation (if diffusers is missing)
    if not HAS_DIFFUSERS:
        print(f"[Dummy] Generating noise video for: '{prompt}'")
        T, H, W, C = 16, height, width, 3
        video = torch.randint(0, 255, (T, H, W, C), dtype=torch.uint8)
        return video

    # 2. Real Model Implementation
    current_seed = base_seed + index
    generator = torch.Generator(device="cuda").manual_seed(current_seed)
    
    print(f"[Generating] Prompt: '{prompt}' | Seed: {current_seed} | Steps: {num_inference_steps} | Res: {width}x{height}")
    
    # Inference
    video_frames = pipe(
        prompt, 
        negative_prompt="bad quality", 
        num_inference_steps=num_inference_steps, 
        height=height,
        width=width,
        generator=generator
    ).frames[0]
    
    # Post-processing: Convert PIL List -> Tensor [T, H, W, C]
    video_np = np.stack([np.array(img) for img in video_frames])
    video_tensor = torch.from_numpy(video_np)
    
    return video_tensor

def main():
    args = parse_args()
    
    if args.seed is not None:
        seed_everything(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for dimension in args.dimensions:
        print(f"Processing dimension: {dimension}")
        
        prompt_file = os.path.join(args.prompts_dir, f"{dimension}.txt")
        if not os.path.exists(prompt_file):
            print(f"Warning: Prompt file not found at {prompt_file}, skipping...")
            continue
            
        with open(prompt_file, 'r') as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        
        dim_save_path = os.path.join(args.save_path, dimension)
        os.makedirs(dim_save_path, exist_ok=True)

        for i, prompt in enumerate(prompt_list):
            print(f"[{i+1}/{len(prompt_list)}] Processing prompt: {prompt}")
            for index in range(args.num_samples):
                filename = f"{prompt}-{index}.mp4"
                file_path = os.path.join(dim_save_path, filename)
                
                if os.path.exists(file_path):
                    print(f"  Skipping existing: {filename}")
                    continue

                video = sample_func(prompt, index, args.seed, args.num_inference_steps, args.flash_attn, args.height, args.width)
                video = process_video_data(video)
                export_to_video(video.transpose(3, 0, 1, 2), file_path, fps=args.fps)

    print("Generation complete.")

if __name__ == "__main__":
    main()
