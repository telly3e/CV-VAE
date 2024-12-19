import os
import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from decord import VideoReader, cpu
from fire import Fire

def evaluate_video_quality(original, noisy):
    """
    Evaluate the quality of the noisy video compared to the original.
    :param original: Original video as a NumPy array (T, H, W, C).
    :param noisy: Noisy video as a NumPy array (T, H, W, C).
    :return: PSNR and SSIM for each frame.
    """
    psnr_values = []
    ssim_values = []

    for i in range(original.shape[0]):
        orig_frame = original[i]
        noisy_frame = noisy[i]

        psnr = peak_signal_noise_ratio(orig_frame, noisy_frame, data_range=255)
        ssim = structural_similarity(orig_frame, noisy_frame, multichannel=True)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    return np.mean(psnr_values), np.mean(ssim_values)

def evaluate_latent_quality(original_latent, noisy_latent):
    """
    Evaluate the quality of noisy latent compared to the original.
    :param original_latent: Original latent tensor.
    :param noisy_latent: Noisy latent tensor.
    :return: MSE and Cosine Similarity.
    """
    mse = torch.mean((original_latent - noisy_latent) ** 2).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        original_latent.flatten(), noisy_latent.flatten(), dim=0
    ).item()

    return mse, cosine_similarity

def load_video_as_numpy(video_path):
    """
    Load a video as a NumPy array in (T, H, W, C) format.
    :param video_path: Path to the video file.
    :return: Video as a NumPy array.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    video = vr.get_batch(range(len(vr))).asnumpy()
    return video  # Shape: (T, H, W, C)

def load_latent(latent_path):
    """
    Load a latent tensor from a .pt file.
    :param latent_path: Path to the latent .pt file.
    :return: Torch tensor containing the latent representation.
    """
    latent = torch.load(latent_path)
    return latent

def main(original_video_dir, noisy_video_dir, original_latent_dir, noisy_latent_dir, latent_video_dir, noisy_latent_video_dir):
    """
    Evaluation
    :param original_video_dir: Directory containing the original videos.
    :param noisy_video_dir: Directory containing the noisy videos.
    :param original_latent_dir: Directory containing the latents of the videos.
    :param noisy_latent_dir: Directory containing the noisy latents of the videos.
    :param latent_video_dir: Directory containing videos generated from latents.
    :param noisy_latent_video_dir: Directory containing noisy videos generated from latents.
    """
    # Iterate over all files in the original video directory
    for video_file in os.listdir(original_video_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_file}")
            continue

        # Construct file paths
        original_video_path = os.path.join(original_video_dir, video_file)
        noisy_video_path = os.path.join(noisy_video_dir, video_file)
        latent_video_path = os.path.join(latent_video_dir, video_file)
        noisy_latent_video_path = os.path.join(noisy_latent_video_dir, video_file)

        # Latent file paths
        original_latent_path = os.path.join(original_latent_dir, video_file.replace('.mp4', '.pt'))
        noisy_latent_path = os.path.join(noisy_latent_dir, video_file.replace('.mp4', '.pt'))

        try:
            # Load videos
            original_video = load_video_as_numpy(original_video_path)
            noisy_video = load_video_as_numpy(noisy_video_path)
            latent_video = load_video_as_numpy(latent_video_path)
            noisy_latent_video = load_video_as_numpy(noisy_latent_video_path)

            # Load latents
            original_latent = load_latent(original_latent_path)
            noisy_latent = load_latent(noisy_latent_path)

            # Video quality evaluation
                # Origin vs Latent
            video_psnr, video_ssim = evaluate_video_quality(original_video, latent_video)
            print(f"[{video_file}] | Origin vs latent | Video PSNR: {video_psnr:.2f}, SSIM: {video_ssim:.3f}")

                # Origin vs Noisy
            video_psnr, video_ssim = evaluate_video_quality(original_video, noisy_video)
            print(f"[{video_file}] | Origin vs Noisy | Video PSNR: {video_psnr:.2f}, SSIM: {video_ssim:.3f}")

                # Origin vs Noisy_latent
            video_psnr, video_ssim = evaluate_video_quality(original_video, noisy_latent_video)
            print(f"[{video_file}] | Origin vs Noisy latent | Video PSNR: {video_psnr:.2f}, SSIM: {video_ssim:.3f}")

                # Noisy vs Noisy_latent (may unnecessary)
            video_psnr, video_ssim = evaluate_video_quality(noisy_video, noisy_latent_video)
            print(f"[{video_file}] | Noisy vs Noisy latent | Video PSNR: {video_psnr:.2f}, SSIM: {video_ssim:.3f}")


            # Latent quality evaluation
            latent_mse, latent_cosine = evaluate_latent_quality(original_latent, noisy_latent)
            print(f"[{video_file}] Latent MSE: {latent_mse:.6f}, Cosine Similarity: {latent_cosine:.3f}")

        except Exception as e:
            print(f"Error processing {video_file}: {e}")

if __name__ == '__main__':
    Fire(main)
