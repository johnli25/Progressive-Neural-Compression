import os
import glob
import subprocess

# Define the directory containing the images
input_dir = "PNC_FrameCorr_input_imgs"  # Replace with the path to your directory
output_dir = "Input_Reconstructed_Videos"  # Directory to store the output videos

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files in the directory
images = glob.glob(os.path.join(input_dir, "*.jpg"))

# Group images by "action_video_number" prefix
file_groups = {}
for img_path in images:
    # Extract the prefix: e.g., "diving_7" from "diving_7_001.jpg"
    prefix = "_".join(os.path.basename(img_path).split("_")[:-1])
    if prefix not in file_groups:
        file_groups[prefix] = []
    file_groups[prefix].append(img_path)

# Sort each group's images by frame number
for prefix in file_groups:
    file_groups[prefix].sort()

print(file_groups.keys())

# Generate videos for each group
for prefix, img_paths in file_groups.items():
    print(prefix, img_paths)
    # Prepare input pattern and output video name
    input_pattern = os.path.join(input_dir, f"{prefix}_%03d.jpg")
    print(input_pattern)
    output_video = os.path.join(output_dir, f"{prefix}.mp4")
    
    # FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        # "-framerate", "30",  # Adjust frame rate if needed
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    
    # Run the command
    print(f"Creating video for {prefix}...")
    subprocess.run(ffmpeg_cmd, check=True)

print(f"All videos saved to {output_dir}")
