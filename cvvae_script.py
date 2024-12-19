from models.modeling_vae import CVVAEModel
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire

def encode_videos(vae_path, video_dir, latent_dir, height=576, width=1024):
    """
    Encode videos in the specified directory to latents and save them.
    :param vae_path: Path to the pretrained VAE model.
    :param video_dir: Directory containing input video files.
    :param latent_dir: Directory to save encoded latent files.
    :param height: Resized height for videos.
    :param width: Resized width for videos.
    """
    vae3d = CVVAEModel.from_pretrained(vae_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False).cuda()

    transform = transforms.Compose([
        transforms.Resize(size=(height, width))
    ])

    os.makedirs(latent_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)

        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_file}")
            continue

        video_reader = VideoReader(video_path, ctx=cpu(0))
        video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
        fps = video_reader.get_avg_fps()

        video = rearrange(torch.tensor(video), 't h w c -> t c h w')
        video = transform(video)
        video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).half()
        frame_end = 1 + (len(video_reader) - 1) // 4 * 4
        video = video / 127.5 - 1.0
        video= video[:,:,:frame_end,:,:]

        latent = vae3d.encode(video.cuda()).latent_dist.sample()
        latent_path = os.path.join(latent_dir, f"{os.path.splitext(video_file)[0]}_latent.pt")

        torch.save(latent.cpu(), latent_path)
        print(f"Encoded and saved latent: {latent_path}")

def decode_latents(vae_path, latent_dir, video_dir, height=576, width=1024):
    """
    Decode latents in the specified directory to videos and save them.
    :param vae_path: Path to the pretrained VAE model.
    :param latent_dir: Directory containing latent files.
    :param video_dir: Directory to save decoded videos.
    :param height: Resized height for videos (used for information only).
    :param width: Resized width for videos (used for information only).
    """
    vae3d = CVVAEModel.from_pretrained(vae_path, subfolder="vae3d", torch_dtype=torch.float16)
    vae3d.requires_grad_(False).cuda()

    os.makedirs(video_dir, exist_ok=True)

    for latent_file in os.listdir(latent_dir):
        if not latent_file.lower().endswith('_latent.pt'):
            print(f"Skipping non-latent file: {latent_file}")
            continue

        latent_path = os.path.join(latent_dir, latent_file)
        latent = torch.load(latent_path).cuda()

        results = vae3d.decode(latent).sample
        results = rearrange(results.squeeze(0), 'c t h w -> t h w c')
        results = (torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5
        results = results.to('cpu', dtype=torch.uint8)

        video_path = os.path.join(video_dir, f"{os.path.splitext(latent_file)[0]}.mp4")
        fps = 30  # Assuming a default FPS, you can adjust if needed.
        write_video(video_path, results, fps=fps, options={'crf': '10'})

        print(f"Decoded and saved video: {video_path}")

if __name__ == '__main__':
    Fire({
        'encode': encode_videos,
        'decode': decode_latents
    })
