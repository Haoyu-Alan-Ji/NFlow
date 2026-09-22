#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess

REPOS = {
    'SS_Group_Shrinkage_New': 'https://github.com/jsanket12/SS_Group_Shrinkage_New.git',
    'spam-pruning': 'https://github.com/fortuinlab/spam-pruning.git',
    'wsBNN': 'https://github.com/AkankshaMishra/wsBNN.git',
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', default='external_methods')
    p.add_argument('--update', action='store_true')
    p.add_argument('--install-spam', action='store_true', help='pip-install SpaM bundled laplace-torch dependencies')
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True, exist_ok=True)
    for name,url in REPOS.items():
        dst=root/name
        if not dst.exists():
            print(f'clone {name}')
            subprocess.run(['git','clone','--depth','1',url,str(dst)], check=True)
        elif a.update:
            print(f'update {name}')
            subprocess.run(['git','-C',str(dst),'pull','--ff-only'], check=True)
        else:
            print(f'keep {name}: {dst}')
    if a.install_spam:
        lap=root/'spam-pruning'/'Laplace_kfac_diag_unitwise'
        print(f'install SpaM Laplace package: {lap}')
        subprocess.run(['python','-m','pip','install','-e',str(lap)], check=True)
if __name__=='__main__': main()
