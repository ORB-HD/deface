#!/usr/bin/env python3

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Directory with videos to be processed')
parser.add_argument('--options', default=None, help='Additional options to be passed to deface.py')

args = parser.parse_args()

cmd_base = 'python3 deface.py'
if args.options is not None:
    cmd_base = f'{cmd_base} {args.options}'

for entry in os.scandir(args.path):
    if entry.is_file():
        os.system(f'{cmd_base} -i {entry.path}')