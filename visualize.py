import numpy as np
import scipy.io as sio
import cv2
import os


def visualize_transient_img():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # --- 1. Define parameters ---
    basedir = os.path.join(project_root, 'data/zaragozadataset')
    db_name = os.path.join(basedir, 'zaragoza256_preprocessed.mat')
    output_name = 'zaragoza256_preprocessed_rep.mp4'
    output_dir = './output_videos'

    # --- 2. Visualizer ---
    # LOAD data
    print(f"Loading data from {db_name}...")
    mat_contents = sio.loadmat(db_name)
    data = mat_contents['data']
    ### data normalization
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 127

    print("Data loaded successfully.")

    video_writer = None
    if output_name:
        os.makedirs(output_dir, exist_ok=True)
        frame_height, frame_width = data.shape[1], data.shape[2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(
            os.path.join(output_dir, output_name),
            fourcc,
            15.0,
            (frame_width, frame_height),
            isColor=False
        )

    num_frames = data.shape[0]
    print(f"Processing {num_frames} frames...")


    for i in range(num_frames):
        # i-th frame data
        I = data[i, :, :]

        I_processed = np.clip(I, 0, 255).astype(np.uint8)


        # Writing
        if video_writer is not None:
            video_writer.write(I_processed)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")

    # Release the instance
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved successfully to: {os.path.join(output_dir, output_name)}")

    print("Process finished.")

if __name__ == '__main__':
    visualize_transient_img()