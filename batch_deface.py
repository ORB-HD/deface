#!/usr/bin/env python3

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Directory with videos to be processed')

args = parser.parse_args()

cmd_base = 'python3 deface.py -m'

for entry in os.scandir(args.path):
    if entry.is_file():
        os.system(f'{cmd_base} -i {entry.path}')