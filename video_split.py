import os
import subprocess
from fire import Fire

def split_video(input_dir, output_dir, segment_duration=10):
    """
    Split long videos into shorter segments of specified duration using FFmpeg.
    :param input_dir: Directory containing input video files.
    :param output_dir: Directory to save split video segments.
    :param segment_duration: Duration of each video segment in seconds (default is 10 seconds).
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_file}")
            continue

        input_path = os.path.join(input_dir, video_file)
        video_name, ext = os.path.splitext(video_file)
        segment_output_template = os.path.join(output_dir, f"{video_name}_%03d{ext}")

        try:
            command = [
                "ffmpeg",
                "-i", input_path,
                "-c", "copy",
                "-map", "0",
                "-f", "segment",
                "-segment_time", str(segment_duration),
                "-reset_timestamps", "1",
                segment_output_template
            ]

            subprocess.run(command, check=True)
            print(f"Successfully split video: {video_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing video {video_file}: {e}")

if __name__ == '__main__':
    Fire(split_video)
