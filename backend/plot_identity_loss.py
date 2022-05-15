
from PIL import Image

import argparse
import os
import cv2
import numpy as np
from losses import cosine_distance

from natsort import natsorted

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.transforms
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)


import torch

import lpips

import blur_detector

loss_fn_vgg = lpips.LPIPS(net='vgg')
young = Image.open('./data/experiment/young3.jpg')
old = Image.open('./data/experiment/old2.jpg')
diff = Image.open('./data/experiment/diff7.jpg')

authentic_loss = cosine_distance(young, old)
diff_loss = cosine_distance(diff, old)
print('authentic_loss ', authentic_loss, 'diff_loss ', diff_loss)
