#!/bin/bash
ffmpeg -f image2 -i vframes/%05d.jpg -r 30 v.mp4 -hide_banner
