import os
import numpy as np

for root, dirs, files in os.walk("logs/magicpoint_synth_homoAdapt_cityscape/predictions/train"):
    for fname in files:
        # print(fname)
        if fname.endswith(".npz"):
            fpath = os.path.join(root, fname)
            # print(fpath)
            try:
                A=np.load(fpath)['pts']
            except Exception as e:
                print("Corrupted", fpath, "|", str(e))
                print(f"[WARN] removing corrupted npz: {fname}")
                # os.remove(fpath)

