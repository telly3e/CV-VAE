import os
import av
from fire import Fire

def split_video(input_dir, output_dir, segment_duration=10):
    """
    Split long videos into shorter segments of specified duration using PyAV.
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

        try:
            container = av.open(input_path)
            output_container = None

            stream = container.streams.video[0]
            stream_time_base = stream.time_base

            segment_start = 0
            segment_index = 0

            for packet in container.demux(stream):
                if packet.dts is None:
                    continue

                packet_time = packet.dts * stream_time_base

                if packet_time >= segment_start + segment_duration:
                    if output_container:
                        output_container.close()

                    segment_start += segment_duration
                    segment_index += 1

                if not output_container or packet_time < segment_start:
                    segment_output_path = os.path.join(output_dir, f"{video_name}_{segment_index:03d}{ext}")
                    output_container = av.open(segment_output_path, mode='w')
                    output_container.add_stream(template=stream)

                output_container.mux(packet)

            if output_container:
                output_container.close()

            print(f"Successfully split video: {video_file}")

        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

if __name__ == '__main__':
    Fire(split_video)
