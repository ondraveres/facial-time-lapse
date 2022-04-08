import traceback
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

from io import BytesIO

from request_boost import boosted_requests

import cv2
import numpy as np
# import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from notebooks.notebook_utils import Downloader, ENCODER_PATHS, INTERFACEGAN_PATHS, STYLECLIP_PATHS
from notebooks.notebook_utils import run_alignment, crop_image, compute_transforms
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, load_encoder, get_average_image
import dex

dex.eval()

print(torch.cuda.current_device(), torch.cuda.get_device_name(0))

experiment_type = 'restyle_pSp_ffhq'
frames_between_images = 15

# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(1024, 1024))


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


n_iters_per_batch = 3  # @param {type:"integer"}
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False  # generate outputs at full resolution

img_transforms = EXPERIMENT_ARGS['transform']


def align_image(pathToImage):
    alligned_im = run_alignment(pathToImage)
    os.remove(pathToImage)
    unique_filename = str(uuid.uuid4())+'.jpg'
    image_path = '../storage/'+unique_filename
    alligned_im.save(image_path)

    age = round(dex.estimate(image_path)[0])
    # img = cv2.imread(image_path)
    # faces = app.get(img)

    # age = faces[0].age

    return unique_filename, age


def invertImage(pathToImage):
    cropped_images = []
    alligned_im = Image.open(pathToImage)
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


def createGif(paths, ages):
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
    pathToGif = f'../storage/timelapse{random.randint(0,100000000)}.gif'
    timelapse_images[0].save(fp=pathToGif, format='GIF', append_images=timelapse_images,
                             save_all=True, duration=80, loop=0)
    return pathToGif


def loss(original_image, generated_image):
    return ((original_image-generated_image)**2).mean()


def saveImagesFromGoogleSearch(phrase, number_of_images):
    phrase1 = f'photo of {phrase} as a kid'
    phrase2 = f'photo of {phrase} as a teenager'
    phrase3 = f'photo of young {phrase}'
    phrase4 = f'photo of {phrase}'

    phrases = [phrase1, phrase2, phrase3, phrase4]
    urls = []
    for phrase in phrases:
        search_term = urllib.parse.quote(phrase.encode('utf8'))
        url = f'https://customsearch.googleapis.com/customsearch/v1?cr=asd&cx=b0aeb7e24cc9a435d&q={search_term}&imgType=face&imgSize=MEDIUM&num={number_of_images}&safe=active&searchType=image&filter=1&key=AIzaSyAxeJGJ-oVB1S5QppevK64MvKWgn7Y-oDU&start=1'
        urls.append(url)

    results = boosted_requests(urls=urls)

    # first_page = requests.get(url_first_page)
    # second_page = requests.get(url_second_page)
    # third_page = requests.get(url_third_page)
    # fourth_page = requests.get(url_fourth_page)

    pathsAndAges = []
    aligned_images = []

    image_urls = []
    headers = []
    for i in range(number_of_images):
        for page in results:
            image_urls.append(page["items"][i]["link"])
            headers.append(
                {'User-Agent': 'Facial time lapse bot/0.0 ondra.veres@gmail.com'})
    print(image_urls)
    image_results = boosted_requests(
        urls=image_urls,  timeout=1, headers=headers, parse_json=False)

    for response in image_results:
        try:
            im = Image.open(BytesIO(response))
            im.convert('RGB').save("temp.jpg", 'jpeg')
            path, age = align_image("temp.jpg")
            left = 412
            top = 412
            right = 612
            bottom = 612
            new_aligned_image = Image.open(
                '../storage/'+path).crop((left, top, right, bottom))
            min_loss = 10000000000000000000000000
            for aligned_image in aligned_images:
                lossv = loss(numpy.asarray(new_aligned_image),
                             numpy.asarray(aligned_image))
                if lossv < min_loss:
                    min_loss = lossv
            if min_loss < 85:
                raise Exception("image is duplicate")

            aligned_images.append(new_aligned_image)

            pathsAndAges.append((path, age))
            done += 1
        except Exception as e:
            print(e)
            i += 1
            if i == number_of_images:
                done = 2
            continue

    # for page in results:
    #     done = 0
    #     i = 0
    #     while done < 1:
    #         try:
    #             print(page["items"][i]["link"])
    #             # headers = {
    #             #     'User-Agent': 'Facial time lapse bot/0.0 ondra.veres@gmail.com',
    #             #     'Transfer-Encoding': 'chunked'
    #             # }
    #             url = page["items"][i]["link"]
    #             response = requests.get(url)

    #             im = Image.open(BytesIO(response.content))
    #             # im = Image.open(requests.get(
    #             #     url, headers=headers, timeout=1).raw)
    #             im.convert('RGB').save("temp.jpg", 'jpeg')
    #             path, age = align_image("temp.jpg")
    #             left = 384
    #             top = 384
    #             right = 384*2
    #             bottom = 384*2
    #             new_aligned_image = Image.open(
    #                 '../storage/'+path).crop((left, top, right, bottom))
    #             min_loss = 10000000000000000000000000
    #             for aligned_image in aligned_images:
    #                 lossv = loss(numpy.asarray(new_aligned_image),
    #                              numpy.asarray(aligned_image))
    #                 if lossv < min_loss:
    #                     min_loss = lossv
    #             if min_loss < 1:
    #                 raise Exception("image is duplicate")

    #             aligned_images.append(new_aligned_image)

    #             pathsAndAges.append((path, age))
    #             done += 1
    #         except Exception as e:
    #             print(e)
    #             i += 1
    #             if i == number_of_images:
    #                 done = 2
    #             continue
    return pathsAndAges
    # return [('hi', 22)]
