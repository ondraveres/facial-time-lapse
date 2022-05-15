import math
from PIL import Image


import argparse
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.transforms
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)

MAX_OPACITY = 0.5

SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['axes.titley'] = 1.02


n = 100
fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(10)
x1 = np.arange(0, n/2, 1)/(n-1)
x2 = np.arange(n/2, n, 1)/(n-1)
x = np.concatenate((x1, x2))
y1 = (np.cos(2*math.pi*x1)*MAX_OPACITY+MAX_OPACITY)/2
y2 = (np.cos(2*math.pi*x2)*MAX_OPACITY+MAX_OPACITY)/2
y = np.concatenate((y1, y2))

ax.fill_between(x1, y1, alpha=0.4, color='tab:orange', edgecolor='none')
ax.fill_between(x2, y2, alpha=0.4, color='tab:olive', edgecolor='none')
ax.fill_between(x, y, 1, alpha=0.4, color='tab:blue', edgecolor='none')
p, = ax.plot([0, 1], [0.5, 0.5], color='tab:grey',
             linestyle='dashed', label=r'$\gamma$', linewidth=2)

plt.xticks(np.arange(11)/10)
# plt.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
ax.set_xlabel(r'Interpolation parameter $\alpha$')
ax.set_ylabel('Blended image composition')
ax.set_title('Image blending')

red_patch = mpatches.Patch(
    color='tab:blue', label=r'Generated image with $\alpha$')
orange_patch = mpatches.Patch(color='tab:orange', label=r'First real image')
olive_patch = mpatches.Patch(color='tab:olive', label=r'Second real image')
plt.legend(handles=[red_patch, orange_patch, olive_patch, p])


# remove tick marks
ax.xaxis.set_tick_params(size=0)
ax.yaxis.set_tick_params(size=0)


# change the color of the top and right spines to opaque gray
ax.spines['right'].set_color((.8, .8, .8))
ax.spines['top'].set_color((.8, .8, .8))

# set the limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# set the grid on
ax.grid('on')


# tweak the axis labels
xlab = ax.xaxis.get_label()
ylab = ax.yaxis.get_label()

xlab.set_style('italic')
# xlab.set_size(14)
ylab.set_style('italic')
# ylab.set_size(14)

# tweak the title
ttl = ax.title
ttl.set_weight('heavy')
ttl.set_size(22)

# plt.subplots_adjust(left=0, bottom=0, right=10, top=10, wspace=0, hspace=0)
plt.savefig('pixel-gott-2-3.pdf')
