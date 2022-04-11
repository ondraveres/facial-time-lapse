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

# # import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from notebooks.notebook_utils import Downloader, ENCODER_PATHS, INTERFACEGAN_PATHS, STYLECLIP_PATHS
from notebooks.notebook_utils import run_alignment, crop_image, compute_transforms
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, load_encoder, get_average_image
import dex

from lib import VGGFace

dex.eval()

print(torch.cuda.current_device(), torch.cuda.get_device_name(0))

experiment_type = 'pSp_stylegan2'
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
    },
    "pSp_stylegan2": {
        "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_toonify.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]


tic = time.time()
model_path3 = "./pretrained_models/restyle_pSp_ffhq.pt"
net3, opts3 = load_encoder(checkpoint_path=model_path3)

model_path2 = "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
net2, opts2 = load_encoder(checkpoint_path=model_path2)
toc = time.time()

print('Loading two models took {:.4f} seconds.'.format(toc - tic))

n_iters_per_batch = 3  # @param {type:"integer"}
opts3.n_iters_per_batch = n_iters_per_batch
opts3.resize_outputs = False  # generate outputs at full resolution

img_transforms = EXPERIMENT_ARGS['transform']

model = VGGFace().double()

model_dict = torch.load('models/vggface.pth',
                        map_location=lambda storage, loc: storage)
model.load_state_dict(model_dict)
# Set model to evaluation mode
model.eval()


def removeOthers(pathsAndAges):
    descriptorsList = []
    for pathAndAge in pathsAndAges:
        img = cv2.imread('../storage/'+pathAndAge[0])
        img = cv2.resize(img, (224, 224))
        # Forward test image through VGGFace
        img = torch.Tensor(img).permute(
            2, 0, 1).view(1, 3, 224, 224).double()
        img -= torch.Tensor(np.array([129.1863, 104.7624,
                            93.5940])).double().view(1, 3, 1, 1)
        descriptor = model(img)[0].detach().numpy()
        descriptorsList.append(descriptor)
        print(descriptor.shape)
    descriptorsArray = numpy.stack(descriptorsList, axis=0)
    descriptor_median = np.median(descriptorsArray, axis=0)
    pathsAndDistances = []
    for i in range(len(descriptorsList)):
        dist = ((descriptorsList[i]-descriptor_median)**2).mean()
        pathsAndDistances.append((pathsAndAges[i][0], dist))

    return pathsAndDistances


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


def invertImage(pathToImage, version):
    alligned_im = Image.open(pathToImage)
    transformed_image = img_transforms(alligned_im)

    if version == 3:
        tensor_with_images = torch.stack([transformed_image], dim=0)
        avg_image = get_average_image(net3)

        with torch.no_grad():
            tic = time.time()
            avg_image_for_batch = avg_image.unsqueeze(
                0).repeat(transformed_image.unsqueeze(
                    0).shape[0], 1, 1, 1)
            x_input = torch.cat(
                [transformed_image.unsqueeze(0).cuda(), avg_image_for_batch], dim=1)

            # _, latent = net3(x_input, return_latents=True, resize=True)

            _, result_latents = run_on_batch(inputs=transformed_image.unsqueeze(0).to(
                "cuda").float(),
                net=net3,
                opts=opts3,
                avg_image=avg_image)

            toc = time.time()
            print('Inference v3 took {:.4f} seconds.'.format(toc - tic))
        # return latent[0]
        return torch.from_numpy(result_latents[0][2])
    if version == 2:
        with torch.no_grad():
            tic = time.time()

            _, latent = net2(transformed_image.unsqueeze(0).to(
                "cuda").float(), return_latents=True, resize=True)
            toc = time.time()
            print('Inference v2 took {:.4f} seconds.'.format(toc - tic))
        return latent


def createGif(paths, ages, version):
    result_latents = []
    for path in paths:
        latent = invertImage(path, version)
        if version == 2:
            tensor_latent = latent.cuda()
        if version == 3:
            tensor_latent = latent.cuda()
        result_latents.append(tensor_latent)
    timelapse_images = []

    with torch.no_grad():

        tic = time.time()
        for i in range(len(result_latents)-1):
            for j in range(frames_between_images):
                t2 = j/(frames_between_images-1)
                print(t2)
                t1 = 1 - t2
                avg = torch.add(torch.mul(result_latents[i], t1), torch.mul(
                    result_latents[i+1], t2))
                mixed_image = None
                if version == 2:
                    mixed_image, _ = net2(
                        avg, return_latents=True, input_code=True, resize=False)
                else:
                    mixed_image, _ = net3(avg.unsqueeze(
                        0), return_latents=True, input_code=True, resize=False)

                timelapse_images.append(
                    tensor2im(mixed_image[0]))
    toc = time.time()
    print(
        'Generating images took {:.4f} seconds.'.format(toc - tic))
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

            pathsAndAges.append(
                ('http://halmos.felk.cvut.cz:5000/storage/'+path, age))
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
