#!/usr/bin/env python3
import argparse
from common import write_json
from ss_common import run
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--repo-root',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--h1',type=int,default=20); p.add_argument('--h2',type=int,default=20); p.add_argument('--device',default='auto'); p.add_argument('--epochs',type=int,default=1200); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--batch',type=int,default=128); p.add_argument('--draws',type=int,default=100); a=p.parse_args(); write_json(run('ss_ghs',a),a.output)
