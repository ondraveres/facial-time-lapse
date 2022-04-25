# import argparse
# import os
# import cv2
# import numpy as np
# from embeddings import get_embeddings

# import torch.utils.data as data
# import torchvision.datasets as datasets
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from backbone import Backbone
from PIL import Image

# from cosinedif import cosine_distance


# def visualize_similarity(tag, input_size=[112, 112]):
#     images, embeddings = get_embeddings(
#         data_root=f"data/{tag}_aligned",
#         model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
#         input_size=input_size,
#     )

#     # calculate cosine similarity matrix
#     print('embedings shape:', embeddings.shape)
#     print('norm is', np.linalg.norm(
#         embeddings[0, :]), 'embedding shape is', embeddings[0, :].shape)
#     cos_similarity = np.dot(embeddings, embeddings.T)
#     print('embedings shape after crazy dot product:',
#           cos_similarity.shape, cos_similarity)
#     cos_similarity = cos_similarity.clip(min=0, max=1)
#     # plot colorful grid from pair distance values in similarity matrix
#     similarity_grid = plot_similarity_grid(cos_similarity, input_size, images)

#     # pad similarity grid with images of faces
#     horizontal_grid = np.hstack(images)
#     vertical_grid = np.vstack(images)
#     zeros = np.zeros((*input_size, 3))
#     vertical_grid = np.vstack((zeros, vertical_grid))
#     result = np.vstack((horizontal_grid, similarity_grid))
#     result = np.hstack((vertical_grid, result))

#     if not os.path.isdir("images"):
#         os.mkdir("images")

#     cv2.imwrite(f"images/{tag}.jpg", result)


# def plot_similarity_grid(cos_similarity, input_size, images):
#     n = len(cos_similarity)
#     rows = []
#     for i in range(n):
#         row = []
#         for j in range(n):
#             # create small colorful image from value in distance matrix
#             value = cos_similarity[i][j]

#             #im1 = Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#             #im2 = Image.fromarray(cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB))
#             #value = cosine_distance(im1, im2)
#             cell = np.empty(input_size)
#             cell.fill(value)
#             cell = (cell * 255).astype(np.uint8)
#             # color depends on value: blue is closer to 0, green is closer to 1
#             img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

#             # add distance value as text centered on image
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             text = f"{value:.2f}"
#             textsize = cv2.getTextSize(text, font, 1, 2)[0]
#             text_x = (img.shape[1] - textsize[0]) // 2
#             text_y = (img.shape[0] + textsize[1]) // 2
#             cv2.putText(
#                 img, text, (text_x, text_y), font, 1, (255,
#                                                        255, 255), 2, cv2.LINE_AA,
#             )
#             row.append(img)
#         rows.append(np.concatenate(row, axis=1))
#     grid = np.concatenate(rows)
#     return grid


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--tags",
#         help="specify your tags for aligned faces datasets",
#         default="test",
#         nargs='+',
#         required=True
#     )
#     args = parser.parse_args()
#     tags = args.tags

#     for tag in tags:
#         visualize_similarity(tag)

import argparse
import os
import cv2
import numpy as np
from losses import cosine_distance

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import matplotlib.transforms
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)

import torch

import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg')


def cv2_to_pil(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image


def load_images_from_folder(folder):
    images = []
    tensors = []
    for filename in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            tensors.append(lpips.im2tensor(
                lpips.load_image(os.path.join(folder, filename))))
            images.append(img)
    return images, tensors


def plot_similarity_grid(original_image, generated_images, cell_size):

    n = len(generated_images)
    rows = []
    row = []
    for i in range(n):
        # create small colorful image from value in distance matrix
        value = cosine_distance(cv2_to_pil(
            original_image), cv2_to_pil(generated_images[i]))

        cell = np.empty([cell_size, cell_size])
        cell.fill(value)
        cell = (cell * 255).astype(np.uint8)
        # color depends on value: blue is closer to 0, green is closer to 1
        img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

        # add distance value as text centered on image
        font = cv2.FONT_HERSHEY_TRIPLEX
        text = f"{value:.2f}"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - textsize[0]) // 2
        text_y = (img.shape[0] + textsize[1]) // 2
        cv2.putText(
            img, text, (text_x, text_y), font, 1, (255,
                                                   255, 255), 2, cv2.LINE_AA,
        )
        row.append(img)
    rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows)

    resized_images = []
    for image in generated_images:
        resized_images.append(cv2.resize(
            image, (112, 112), interpolation=cv2.INTER_LINEAR))

    resized_original_image = cv2.resize(
        original_image, (112, 112), interpolation=cv2.INTER_LINEAR)

    horizontal_grid = np.hstack(resized_images)
    vertical_grid = np.vstack([resized_original_image])
    zeros = np.zeros((*[cell_size, cell_size], 3))
    vertical_grid = np.vstack((zeros, vertical_grid))
    result = np.vstack((horizontal_grid, grid))
    result = np.hstack((vertical_grid, result))

    if not os.path.isdir("images"):
        os.mkdir("images")

    cv2.imwrite(f"images/experiment.jpg", result)


def pixel_loss(image1, image2):
    print((np.array(image2)/255)[50, 50])
    return (((np.array(image1)/255)-(np.array(image2)/255))**2).mean()

# original_image = cv2.imread('experiments/original_image.jpg')
# compared_images = load_images_from_folder('experiments/compared_images')

# plot_similarity_grid(original_image, compared_images, 112)


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

plt.rcParams['axes.titley'] = 1.04

original_image_tensor = lpips.im2tensor(lpips.load_image('data/pixel/0.jpg'))
original_image = Image.open('data/pixel/0.jpg')
compared_images, compared_images_tensors = load_images_from_folder(
    'data/restyle0')

n = len(compared_images)
displayed_images = 6

pixel_losses = np.zeros(n)
id_losses = np.zeros(n)
l_pips_losses = np.zeros(n)
pixel_losses = np.random.rand(n)

for index, generated_img in enumerate(compared_images):
    pixel_losses[index] = pixel_loss(original_image, generated_img)
    id_losses[index] = 1-cosine_distance(original_image, generated_img)
    l_pips_losses[index] = loss_fn_vgg(
        original_image_tensor, compared_images_tensors[index])

fig, ax = plt.subplots()
fig.set_figheight(6.25)
fig.set_figwidth(n*1.3 + 3)
x = np.arange(0, n, 1)/(n-1)
y = np.random.rand(n)

ax.plot(x, pixel_losses, label="Pixel-wise Loss")
ax.plot(x, id_losses, label="Identity Loss")
ax.plot(x, l_pips_losses, label="Perceptual Loss")


for index, anotation_img in enumerate(compared_images):
    imagebox = OffsetImage(anotation_img, zoom=0.12)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, (index/(displayed_images-1), -0.08),
                        xybox=(0, -7),
                        pad=0.2,
                        xycoords=("data", "axes fraction"),
                        boxcoords="offset points",
                        box_alignment=(.5, 1),
                        bboxprops={"edgecolor": "none"})

    ax.add_artist(ab)

plt.xticks(np.arange(displayed_images)/(displayed_images-1),
           [r'$0$', r'$1/5$', r'$2/5$', r'$3/5$', r'$4/5$', r'$1$'])
plt.legend()
plt.subplots_adjust(top=0.9, bottom=0.4, left=0.1, right=0.9)
ax.set_xlabel(r'Interpolation parameter $\alpha$')
ax.set_ylabel('Loss Value')
ax.set_title('StyleGAN3 (ReStyle pSp) without image blending evaluation')

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

# move x label down
ax.xaxis.set_label_coords(0.45, -0.70)
# apply offset transform to all x ticklabels.
dx = 0/72.
dy = -0.05  # -1.35
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

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
