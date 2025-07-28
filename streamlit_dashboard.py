import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob
import cv2
import matplotlib.colors as mcolors
from PIL import Image

# --- Config ---
BASE_LOG_DIR = 'runs/'
NPZ_SEARCH_PATH = 'logs/'

TAG_DESCRIPTIONS = {
    'num_points': 'Number of keypoints detected',
    'train-loss': 'Total training loss',
    'train-seg_loss': 'Segmentation loss during training',
    'train-precision': 'Keypoint precision (training)',
    'train-recall': 'Keypoint recall (training)',
    'val-loss': 'Total validation loss',
    'val-seg_loss': 'Segmentation loss during validation',
    'val-precision': 'Keypoint precision (validation)',
    'val-recall': 'Keypoint recall (validation)',
}

ALL_SCALAR_TAGS = list(TAG_DESCRIPTIONS.keys()) + [
    'train-loss_det', 'train-loss_det_warp', 'train-negative_dist', 'train-positive_dist',
    'train-original_gt_resi_var_x', 'train-original_gt_resi_var_y',
    'train-warped_gt_resi_var_x', 'train-warped_gt_resi_var_y',
    'val-loss_det', 'val-loss_det_warp', 'val-negative_dist', 'val-positive_dist',
    'val-original_gt_resi_var_x', 'val-original_gt_resi_var_y',
    'val-warped_gt_resi_var_x', 'val-warped_gt_resi_var_y'
]

IMPORTANT_TAGS = list(TAG_DESCRIPTIONS.keys())

LEGEND_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

def list_all_experiment_folders(base_dir):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                experiment_dirs.append(root)
                break
    return sorted(set(os.path.relpath(d, base_dir) for d in experiment_dirs))

def load_tensorboard_scalars(log_dir, scalar_tags):
    scalars = {tag: [] for tag in scalar_tags}
    steps = []
    tf_event_files = glob(os.path.join(log_dir, "events.out.tfevents.*"))
    for file in tf_event_files:
        ea = EventAccumulator(file)
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        if not any(tag in available_tags for tag in scalar_tags):
            continue
        if not steps:
            # Use the first scalar tag that exists in this file to initialize steps
            for t in scalar_tags:
                if t in available_tags:
                    steps = [s.step for s in ea.Scalars(t)]
                    break
        for tag in scalar_tags:
            if tag in available_tags:
                scalars[tag] = [s.value for s in ea.Scalars(tag)]
    return steps, scalars

def colorize_mask(mask, num_classes):
    color_map = np.array([mcolors.to_rgb(LEGEND_COLORS[i % len(LEGEND_COLORS)]) for i in range(num_classes)]) * 255
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        mask_rgb[mask == i] = color_map[i]
    return mask_rgb.astype(np.uint8), color_map.astype(np.uint8)

def overlay_keypoints(image, keypoints, color=(0, 255, 0)):
    img = image.copy()
    for pt in keypoints:
        if len(pt) == 2:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)
    return img

def load_npz_images(base_path, selected_folder):
    folder_path = os.path.join(base_path, selected_folder)
    npz_files = glob(os.path.join(folder_path, '**', '*.npz'), recursive=True)
    overlays = []
    for f in sorted(npz_files):
        try:
            data = np.load(f)
            base_img = data.get('image') or data.get('image1')
            pred_mask = data.get('pred_mask') or data.get('segmentation_mask')
            gt_mask = data.get('gt_mask') or data.get('mask_gt') or data.get('segmentation_gt')
            conf = data.get('confidence')
            kpts = data.get('keypoints')
            if base_img is None or pred_mask is None:
                continue
            base_img = base_img.astype(np.uint8)
            if base_img.ndim == 2:
                base_img = np.stack([base_img]*3, axis=-1)
            if pred_mask.ndim != 2:
                continue
            num_classes = int(max(pred_mask.max(), gt_mask.max() if gt_mask is not None else 0)) + 1
            pred_color, legend_map = colorize_mask(pred_mask, num_classes)
            pred_overlay = cv2.addWeighted(base_img, 0.5, pred_color, 0.5, 0)
            if conf is not None:
                heatmap = (conf * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                pred_overlay = cv2.addWeighted(pred_overlay, 0.6, heatmap, 0.4, 0)
            if kpts is not None:
                pred_overlay = overlay_keypoints(pred_overlay, kpts)
            if gt_mask is not None:
                gt_color, _ = colorize_mask(gt_mask, num_classes)
                gt_overlay = cv2.addWeighted(base_img, 0.5, gt_color, 0.5, 0)
                comparison = np.concatenate([gt_overlay, pred_overlay], axis=1)
                overlays.append((f, comparison, legend_map))
            else:
                overlays.append((f, pred_overlay, legend_map))
        except Exception as e:
            continue
    return overlays

st.title("📊 SuperPoint Training Dashboard")

with st.sidebar:
    st.header("Settings")
    experiments = list_all_experiment_folders(BASE_LOG_DIR)
    experiment_name = st.selectbox("Select experiment folder", experiments)
    event_files = glob(os.path.join(os.path.join(BASE_LOG_DIR, experiment_name), "events.out.tfevents.*"))
    if event_files:
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        available_scalar_tags = ea.Tags().get('scalars', [])
    else:
        available_scalar_tags = []

    valid_scalar_tags = [t for t in ALL_SCALAR_TAGS if t in available_scalar_tags]
    scalar_tags = st.multiselect(
        "Select scalars to visualize",
        options=valid_scalar_tags,
        default=[t for t in IMPORTANT_TAGS if t in valid_scalar_tags],
        format_func=lambda tag: f"{tag} - {TAG_DESCRIPTIONS.get(tag, 'No description')}"
    )
    show_npz_overlay = st.checkbox("Show .npz overlay previews from logs/", value=True)
    if show_npz_overlay:
        folder_candidates = sorted([f for f in os.listdir(NPZ_SEARCH_PATH) if os.path.isdir(os.path.join(NPZ_SEARCH_PATH, f))])
        selected_npz_folder = st.selectbox("Select subfolder from logs/", folder_candidates)

if experiment_name:
    log_path = os.path.join(BASE_LOG_DIR, experiment_name)

    if scalar_tags:
        with st.spinner(f"Loading logs from {experiment_name}..."):
            steps, scalars = load_tensorboard_scalars(log_path, scalar_tags)

        if steps:
            grouped = {'Loss': [], 'Accuracy': [], 'Other': []}
            for tag in scalar_tags:
                if 'loss' in tag:
                    grouped['Loss'].append(tag)
                elif 'precision' in tag or 'recall' in tag:
                    grouped['Accuracy'].append(tag)
                else:
                    grouped['Other'].append(tag)

            for group_name, tags in grouped.items():
                if tags:
                    st.markdown(f"### {group_name} Metrics")
                    for tag in tags:
                        if tag in scalars:
                            st.subheader(f"{tag}")
                            fig, ax = plt.subplots()
                            ax.plot(steps[:len(scalars[tag])], scalars[tag])
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel(tag)
                            st.pyplot(fig)
                            # Close figure after rendering to avoid memory leaks
                            plt.close(fig)
        else:
            st.warning("No valid data found in TensorBoard logs.")

    if show_npz_overlay and selected_npz_folder:
        st.markdown(f"### Overlay Visuals from logs/{selected_npz_folder}")
        overlays = load_npz_images(NPZ_SEARCH_PATH, selected_npz_folder)
        if overlays:
            for f, img, legend_map in overlays[:10]:
                st.image(img, caption=os.path.relpath(f, NPZ_SEARCH_PATH), channels="RGB", use_column_width='auto')
                if legend_map is not None:
                    st.markdown("**Class Color Legend:**")
                    for idx, color in enumerate(legend_map):
                        color_hex = '#%02x%02x%02x' % tuple(color)
                        st.markdown(f"<span style='display:inline-block;width:15px;height:15px;background-color:{color_hex};margin-right:10px'></span> Class {idx}", unsafe_allow_html=True)
                st.download_button("Download Image", data=Image.fromarray(img).tobytes(), file_name=os.path.basename(f) + ".png", mime="image/png")
        else:
            st.info("No .npz prediction overlays found in this folder.")
else:
    st.info("Use the sidebar to select an experiment and scalar tags.")
# This code is a Streamlit dashboard for visualizing training metrics and overlays from SuperPoint experiments.
# It allows users to select an experiment folder, view scalar metrics from TensorBoard logs, and optionally display overlays from .npz files containing keypoint predictions and segmentation masks.
# The dashboard includes features for dynamically loading available scalar tags, visualizing keypoint detection metrics, and displaying image overlays with segmentation masks and keypoints.
# The user can also download the displayed images directly from the dashboard.