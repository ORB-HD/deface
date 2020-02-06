#!/usr/bin/env python3

import argparse
import os
import glob
import sys
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Directory with videos to be processed')
parser.add_argument('-e', '--ext', default='*', help='Filter by file extension (no filter (*) by default)')
parser.add_argument('--options', default=None, help='Additional options to be passed to deface.py')

args = parser.parse_args()

cmd_base = f'{sys.executable} deface.py -q --nested'
if args.options is not None:
    cmd_base = f'{cmd_base} {args.options}'
# cmd_base = f'{cmd_base}'

paths = glob.glob(f'{args.path}/**/*.{args.ext}', recursive=True)

pbar = tqdm.tqdm(paths, position=0)

for p in pbar:
    pbar.set_description(f'Current video: {p}')
    os.system(f'{cmd_base} -i {p}')