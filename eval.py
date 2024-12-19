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

def main(original_video_path,noisy_video_path,original_latent_path,noisy_latent_path):
    """
    Evaluatation
    :param original_video_path: Directory containing the original video.
    :param noisy_video_path: Directory containing the noisy video.
    :param original_latent_path: Directory containing the latent of the video.
    :param noisy_latent_path: Directory containing the noisy latent of the video.
    """
    original_video = load_video_as_numpy(original_video_path)  # 原始视频加载为 NumPy 数组 (T, H, W, C)
    noisy_video = load_video_as_numpy(noisy_video_path)     # 加噪后的视频加载为 NumPy 数组 (T, H, W, C)
    original_latent = load_latent(original_latent_path) # 原始潜在变量加载为 Torch 张量
    noisy_latent = load_latent(noisy_latent_path)    # 加噪后的潜在变量加载为 Torch 张量

    # 视频质量评估
    video_psnr, video_ssim = evaluate_video_quality(original_video, noisy_video)
    print(f"Video PSNR: {video_psnr:.2f}, SSIM: {video_ssim:.3f}")

    # 潜在变量质量评估
    latent_mse, latent_cosine = evaluate_latent_quality(original_latent, noisy_latent)
    print(f"Latent MSE: {latent_mse:.6f}, Cosine Similarity: {latent_cosine:.3f}")

if __name__ == '__main__':
    Fire(main)