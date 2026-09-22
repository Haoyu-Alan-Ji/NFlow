#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess
REPO='https://github.com/jsanket12/SS_Group_Shrinkage_New.git'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='external_methods'); p.add_argument('--update',action='store_true'); a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True); dst=root/'SS_Group_Shrinkage_New'
    if not dst.exists(): subprocess.run(['git','clone','--depth','1',REPO,str(dst)],check=True)
    elif a.update: subprocess.run(['git','-C',str(dst),'pull','--ff-only'],check=True)
    else: print('keep',dst)
if __name__=='__main__': main()
