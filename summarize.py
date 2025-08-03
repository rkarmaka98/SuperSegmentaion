import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

def load_metrics(folder):
    """
    Load result.npz from folder.
    Expects folder/result.npz with keys:
      'repeatability', 'localization_err', 'correctness', 
      'homography_thresh', 'mscore', 'mAP'
    """
    path = os.path.join(folder, "result.npz")
    if not os.path.isfile(path):
        print(f"  Warning: no result.npz in {folder}; skipping")
        return None
    data = np.load(path, allow_pickle=True)
    return data

def summarize(data):
    out = {}
    # Repeatability
    rep = data.get('repeatability')
    if rep is None or len(rep)==0:
        out['Repeatability'] = np.nan
    else:
        out['Repeatability'] = float(np.mean(rep))
    # MLE (mean localization error)
    loc = data.get('localization_err')
    if loc is None or len(loc)==0:
        out['MLE'] = np.nan
    else:
        out['MLE'] = float(np.mean(loc))
    # NN mAP
    mAP = data.get('mAP')
    if mAP is None or len(mAP)==0:
        out['NN mAP'] = np.nan
    else:
        out['NN mAP'] = float(np.mean(mAP))
    # Matching Score
    msc = data.get('mscore')
    if msc is None or len(msc)==0:
        out['Matching Score'] = np.nan
    else:
        out['Matching Score'] = float(np.mean(msc))
    # Homography@1,3,5
    corr = data.get('correctness')
    thresh = data.get('homography_thresh')
    for eps in (1,3,5):
        key = f"Homography@{eps}"
        if corr is None or thresh is None:
            out[key] = np.nan
        else:
            arr = np.array(thresh)
            idxs = np.where(arr == eps)[0]
            if len(idxs)==0:
                out[key] = np.nan
            else:
                i = idxs[0]
                corr_arr = np.array(corr, dtype=float)
                # corr_arr shape: (num_files, num_thresholds)
                if corr_arr.size==0 or corr_arr.ndim<2 or i>=corr_arr.shape[1]:
                    out[key] = np.nan
                else:
                    out[key] = float(np.mean(corr_arr[:, i]))
    
    # Segmentation IOU
    seg_iou = data.get('segmentation_iou')
    if seg_iou is None or len(seg_iou)==0:
        out['Segmentation IOU'] = np.nan
    else:
        out['Segmentation IOU'] = float(np.mean(seg_iou))
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <folder1> [folder2 ...]")
        sys.exit(1)
    folders = sys.argv[1:]
    rows = []
    for folder in folders:
        # derive task name as parent folder name
        p = Path(folder)
        if p.name.lower() == "predictions" and p.parent.name:
            name = p.parent.name
        else:
            # otherwise fallback to folder name
            name = p.name
        data = load_metrics(folder)
        if data is None:
            # skip or add NaN row
            row = {'Task': name,
                    'Homography@3': np.nan, 'Homography@5': np.nan,
                   'Repeatability': np.nan, 'MLE': np.nan,
                   'NN mAP': np.nan, 'Matching Score': np.nan, 'Segmentation IOU': np.nan}
        else:
            summ = summarize(data)
            summ['Task'] = name
            row = summ
        rows.append(row)
    df = pd.DataFrame(rows, columns=[
        'Task', 'Homography@3', 'Homography@5',
        'Repeatability', 'MLE', 'NN mAP', 'Matching Score', 'Segmentation IOU'
    ])
    # Print table
    print(df.to_string(index=False))
    # Save CSV
    out_csv = "summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()
