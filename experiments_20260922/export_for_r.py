#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def w(path,x): np.savetxt(path,x,delimiter=",")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out"); a=ap.parse_args(); z=np.load(a.data,allow_pickle=False); tr=z["train_idx"].astype(int); te=z["test_idx"].astype(int); o=Path(a.out); o.mkdir(parents=True,exist_ok=True)
    w(o/"X_train.csv",z["X"][tr]); w(o/"y_train.csv",z["y"][tr]); w(o/"X_test.csv",z["X"][te]); w(o/"y_test.csv",z["y"][te]); w(o/"signal_test.csv",z["signal"][te]); w(o/"feature_true.csv",z["feature_true"])
    (o/"meta.json").write_text(json.dumps({"seed":int(z["seed"].item()),"condition":str(z["condition"].item())},indent=2),encoding="utf-8")
if __name__=="__main__": main()
