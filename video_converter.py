# video_converter.py
import tempfile
import subprocess
import os

def convert_to_avi(input_path):
    """
    Converts any video file to .avi using ffmpeg
    Returns path to the converted file
    """
    base_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(tempfile.gettempdir(), f"{base_name}.avi")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_path
