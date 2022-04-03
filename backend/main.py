import numpy
import cv2
import matplotlib.pyplot as plt
import requests
import time
import sys
import os
import pprint
import numpy as np
from PIL import Image
import dataclasses
import torch
import torchvision.transforms as transforms
import urllib
import random
import uuid

import cv2
import numpy as np
#import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from notebooks.notebook_utils import Downloader, ENCODER_PATHS, INTERFACEGAN_PATHS, STYLECLIP_PATHS
from notebooks.notebook_utils import run_alignment, crop_image, compute_transforms
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, load_encoder, get_average_image

print(torch.cuda.current_device(), torch.cuda.get_device_name(0))

experiment_type = 'restyle_pSp_ffhq'
frames_between_images = 15

#app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#app.prepare(ctx_id=0, det_size=(640, 640))


EXPERIMENT_DATA_ARGS = {
    "restyle_pSp_ffhq": {
        "model_path": "./pretrained_models/restyle_pSp_ffhq.pt",
        "image_path": "./notebooks/images/face_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "restyle_e4e_ffhq": {
        "model_path": "./pretrained_models/restyle_e4e_ffhq.pt",
        "image_path": "./notebooks/images/face_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

download_with_pydrive = False  # @param {type:"boolean"}
downloader = Downloader(code_dir='facial-time-lapse-video',
                        use_pydrive=download_with_pydrive,
                        subdir="pretrained_models")


# @title Load ReStyle SG3 Encoder { display-mode: "form" }
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_encoder(checkpoint_path=model_path)
pprint.pprint(dataclasses.asdict(opts))


n_iters_per_batch = 3  # @param {type:"integer"}
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False  # generate outputs at full resolution

img_transforms = EXPERIMENT_ARGS['transform']


def align_image(pathToImage):
    alligned_im = run_alignment(pathToImage)
    unique_filename = str(uuid.uuid4())+'.jpg'
    image_path = '../storage/'+unique_filename
    alligned_im.save(image_path)

    #img = cv2.imread(image_path)
    #faces = app.get(img)

    #age = faces[0].age

    return unique_filename, random.randint(3, 80)


def invertImage(pathToImage):
    cropped_images = []
    alligned_im = run_alignment(pathToImage)
    cropped_images.append(alligned_im)

    transformed_images = []
    for cropped_image in cropped_images:
        transformed_images.append(img_transforms(cropped_image))
        # transformed_images.append(img_transforms(cropped_image))
        # transformed_images.append(img_transforms(cropped_image))
        # transformed_images.append(img_transforms(cropped_image))

    tensor_with_images = torch.stack(transformed_images, dim=0)
    avg_image = get_average_image(net)

    with torch.no_grad():
        tic = time.time()

        result_batch, result_latents = run_on_batch(inputs=tensor_with_images.cuda().float(),
                                                    net=net,
                                                    opts=opts,
                                                    avg_image=avg_image)
        # landmarks_transform=torch.from_numpy(landmarks_transform).cuda().float())

        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    result_latent = result_latents[0][2]
    # result_tensors = result_batch[0]  # there's one image in our batch
    # resize_amount = (256, 256) if opts.resize_outputs else (
    #    opts.output_size, opts.output_size)
    # final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)

    # res = Image.fromarray(np.array(final_rec))
    # os.remove(pathToImage)
    return result_latent


def createGif(paths):
    result_latents = []
    for path in paths:
        result_latents.append(invertImage(path))
    timelapse_images = []
    with torch.no_grad():
        for i in range(len(result_latents)-1):
            print('i is', i)
            for j in range(frames_between_images):
                t2 = j/(frames_between_images-1)
                t1 = 1 - t2
                avg = result_latents[i] * t1 + result_latents[i+1]*t2
                mixed_image = net(torch.from_numpy(avg).cuda().unsqueeze(
                    0), return_latents=False, input_code=True, resize=False)
                # tensor = torch.from_numpy(avg).unsqueeze(0)
                # mixed_image, result_latent = net(torch.cat((tensor, tensor, tensor), 0).cuda(
                # ), return_latents=True, input_code=True, resize=False)

                timelapse_images.append(Image.fromarray(
                    np.array(tensor2im(mixed_image[0])).astype(np.uint8)))
    pathToGif = f'timelapse{random.randint(0,100000000)}.gif'
    timelapse_images[0].save(fp=pathToGif, format='GIF', append_images=timelapse_images,
                             save_all=True, duration=80, loop=0)
    return pathToGif


def loss(original_image, generated_image):
    return ((original_image-generated_image)**2).mean()


def saveImagesFromGoogleSearch(phrase, number_of_images):
    cropped_images = []
    image_paths = []
    search_term = urllib.parse.quote(phrase.encode('utf8'))

    url_first_page = f'https://customsearch.googleapis.com/customsearch/v1?cr=asd&cx=b0aeb7e24cc9a435d&exactTerms={search_term}&imgType=face&num={number_of_images}&safe=active&searchType=image&filter=1&key=AIzaSyCwBdmXc5vg4GC3dEK2iDZC0kuXFjKe6-U&start=1'
    url_second_page = f'https://customsearch.googleapis.com/customsearch/v1?cr=asd&cx=b0aeb7e24cc9a435d&exactTerms={search_term}&imgType=face&num={number_of_images}&safe=active&searchType=image&filter=1&key=AIzaSyCwBdmXc5vg4GC3dEK2iDZC0kuXFjKe6-U&start=21'
    first_page = requests.get(url_first_page)
    second_page = requests.get(url_second_page)

    images = []

    c = 0
    for page in [first_page, second_page]:
        for i in range(number_of_images):
            try:
                if len(cropped_images) > 4:
                    continue
                im = Image.open(requests.get(page.json()["items"][i]["link"], headers={
                                'User-Agent': 'Facial time lapse bot/0.0 ondra.veres@gmail.com'}, stream=True).raw)
                im.save(f"{c}.jpg")
                alligned_im = run_alignment(f"{c}.jpg")
                min_loss = 10000000000000000000000000
                for cropped_image in cropped_images:
                    lossv = loss(numpy.asarray(alligned_im),
                                 numpy.asarray(cropped_image))
                    if lossv < min_loss:
                        min_loss = lossv
                if min_loss < 70:
                    continue
                cropped_images.append(alligned_im)
                image_paths.append(f"{c}.jpg")
                print("min loss is", min_loss)
                images.append(im)
                c += 1
                print(im.size[0]/im.size[1])
                # display(im.resize((256,256)))
            except Exception as e:
                print(e)
                c -= 1
                continue
    return image_paths
