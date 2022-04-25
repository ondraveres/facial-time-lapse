import numpy as np
from align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from align.detector import detect_faces
from PIL import Image

import torch

from torchvision import transforms

import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone import Backbone
from tqdm import tqdm
import os

model_root = "checkpoint/backbone_ir50_ms1m_epoch120.pth"

reference = get_reference_facial_points(default_square=True)
crop_size = 112
input_size = [112, 112]

transform = transforms.Compose(
    [
        transforms.Resize(
            [int(128 * input_size[0] / 112),
             int(128 * input_size[0] / 112)],
        ),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load backbone weigths from a checkpoint
backbone = Backbone(input_size)
backbone.load_state_dict(torch.load(
    model_root, map_location=torch.device("cpu")))
backbone.to(device)
backbone.eval()


def cosine_distance(image1, image2):

    embeddings = []
    for img in [image1, image2]:
        _, landmarks = detect_faces(img)

        if (len(landmarks) == 0):
            print("{} is discarded due to non-detected landmarks!")
            continue
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]]
                         for j in range(5)]
        warped_face = warp_and_crop_face(
            np.array(img),
            facial5points,
            reference,
            crop_size=(crop_size, crop_size),
        )
        img_warped = Image.fromarray(warped_face)

        image_tensor = transform(img_warped).unsqueeze(0)

        embedding = np.zeros([512])
        with torch.no_grad():
            embedding[:] = F.normalize(
                backbone(image_tensor.to(device))).cpu()
            print('first item in embeddings is', embedding[0])
            embeddings.append(embedding)

    cosine_dist = np.dot(embeddings[0], embeddings[1].T)
    return cosine_dist
