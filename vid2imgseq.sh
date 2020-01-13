#!/bin/bash
mkdir -p vframes;
FRAMERATE=$(ffmpeg -i $1 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")
ffmpeg -i $1 -qscale:v 2 vframes/%05d.jpg -hide_banner
