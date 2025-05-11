import os
import cv2
import numpy as np
from scipy.io import loadmat

def convert_mat_to_jpg(mat_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(mat_dir):
        if not fname.endswith('.mat'):
            continue
        base = os.path.splitext(fname)[0]
        full_path = os.path.join(mat_dir, fname)

        try:
            mat = loadmat(full_path)
            ground_truths = mat['groundTruth']
            count = ground_truths.shape[0] if ground_truths.shape[0] > 1 else ground_truths.shape[1]

            for i in range(count):
                entry = ground_truths[i][0] if ground_truths.shape[1] == 1 else ground_truths[0][i]
                boundaries = entry['Boundaries'][0, 0]  # shape HxW
                binary_mask = (boundaries > 0).astype(np.uint8) * 255

                out_path = os.path.join(out_dir, f"{base}_{i}.jpg")
                cv2.imwrite(out_path, binary_mask)
            print(f"Converted {fname} → {count} mask(s)")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

# Example usage
if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        in_dir = f"archive/ground_truth/{split}"
        out_dir = f"archive/converted_ground_truth/{split}"
        convert_mat_to_jpg(in_dir, out_dir)
