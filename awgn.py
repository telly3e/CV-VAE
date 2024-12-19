import torch
import numpy as np
import os
from decord import VideoReader, cpu
from einops import rearrange
from torchvision.io import write_video
from fire import Fire

def add_awgn(data, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to data.
    :param data: Input tensor (latent or video).
    :param snr_db: Signal-to-Noise Ratio in dB.
    :return: Noisy tensor.
    """
    signal_power = torch.mean(data ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_std = torch.sqrt(signal_power / snr_linear)
    noise = noise_std * torch.randn_like(data)
    return data + noise

def simulate_latent_awgn(latent_dir, output_dir, snr_db):
    """
    Apply AWGN to all latent tensors in the specified directory.
    :param latent_dir: Directory containing latent tensors.
    :param output_dir: Directory to save noisy latents.
    :param snr_db: Signal-to-Noise Ratio in dB.
    """
    os.makedirs(output_dir, exist_ok=True)

    for latent_file in os.listdir(latent_dir):
        if not latent_file.endswith(".pt"):
            print(f"Skipping non-latent file: {latent_file}")
            continue

        latent_path = os.path.join(latent_dir, latent_file)
        latent = torch.load(latent_path).float() # pytorch does not support float sqrt in cpu
        noisy_latent = add_awgn(latent, snr_db).half() # float -> half

        noisy_latent_path = os.path.join(output_dir, latent_file)
        torch.save(noisy_latent, noisy_latent_path)
        print(f"Saved noisy latent: {noisy_latent_path}")

def simulate_video_awgn(video_dir, output_dir, snr_db):
    """
    Apply AWGN to all videos in the specified directory.
    :param video_dir: Directory containing video files.
    :param output_dir: Directory to save noisy videos.
    :param snr_db: Signal-to-Noise Ratio in dB.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_file}")
            continue

        video_path = os.path.join(video_dir, video_file)
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
        fps = video_reader.get_avg_fps()

        video = rearrange(torch.tensor(video), 't h w c -> t c h w')
        video = video / 127.5 - 1.0  # Normalize to [-1, 1]
        video.float() # pytorch does not support float sqrt in cpu
        noisy_video = add_awgn(video, snr_db)
        noisy_video = (torch.clamp(noisy_video, -1.0, 1.0) + 1.0) * 127.5
        noisy_video = noisy_video.to('cpu', dtype=torch.uint8)
        noisy_video = rearrange(noisy_video, 't c h w -> t h w c')

        noisy_video_path = os.path.join(output_dir, video_file)
        write_video(noisy_video_path, noisy_video, fps=fps, options={'crf': '10'})
        print(f"Saved noisy video: {noisy_video_path}")

if __name__ == '__main__':
    Fire({
        'simulate_latent': simulate_latent_awgn,
        'simulate_video': simulate_video_awgn
    })
