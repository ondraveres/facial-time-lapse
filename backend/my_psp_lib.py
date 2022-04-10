
from pixel2style2pixel.scripts.align_all_parallel import align_face
import dlib
from pixel2style2pixel.models.psp import pSp
from pixel2style2pixel.utils.common import tensor2im, log_input_image
from pixel2style2pixel.datasets import augmentations
import os

from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import imageio

sys.path.append(".")
sys.path.append("..")


def transform(x):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(x)


def transform_high_res(x):
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(x)


def run_experiment(image_paths, frames_between_images, frames_pixel_interpolation, experiment_type):

    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",
        },
        "ffhq_frontalize": {
            "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_frontalization.pt",
        },
        "toonify": {
            "model_path": "pixel2style2pixel/pretrained_models/psp_ffhq_toonify.pt",
        },
    }

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
        raise ValueError(
            "Pretrained model was unable to be downlaoded correctly!")

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    pprint.pprint(opts)

    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    transformed_images = []
    i = 0
    for image_path in image_paths:
        i += 1
        input_image = run_alignment(image_path, 256)
        input_image.save(f"original{i}.jpg")
        transformed_image = transform(input_image)
        transformed_images.append(transformed_image)

    # transformed_images_high_res = []
    # for image_path in image_paths:
    #     input_image = run_alignment(image_path,1024)
    #     transformed_image = transform_high_res(input_image)
    #     transformed_images_high_res.append(transformed_image)

    with torch.no_grad():
        tic = time.time()
        latents = []
        result_images = []
        i = 0
        for transformed_image in transformed_images:
            i += 1
            image, latent = net(transformed_image.unsqueeze(0).to(
                "cuda").float(), return_latents=True, resize=True)
            latents.append(latent)
            Image.fromarray(np.array(tensor2im(image[0])).astype(np.uint8)).save(f"{i}.jpg")
        for i in range(len(latents)-1):
            print('i is', i)
            finalInversion = None
            for j in range(frames_between_images):
                t2 = j/(frames_between_images-1)
                t1 = 1 - t2
                avg = latents[i] * t1 + latents[i+1]*t2
                mixed_image, result_latent = net(
                    avg, return_latents=True, input_code=True, resize=True)
                # if j<frames_pixel_interpolation:
                #     c2 = (j/(frames_pixel_interpolation-1))
                #     c1 = 1- c2
                #     mix = transformed_images[i].cuda()* c1 + mixed_image.cuda()*c2
                #     result_images.append(Image.fromarray(np.array(tensor2im(mix[0])).astype(np.uint8)))
                # elif j>=frames_between_images-frames_pixel_interpolation:
                #     c2 = ((j-(frames_between_images-frames_pixel_interpolation))/(frames_pixel_interpolation-1))*1
                #     c1 = 1- c2
                #     mix = mixed_image.cuda()* c1 +transformed_images[i+1].cuda()*c2
                #     result_images.append(Image.fromarray(np.array(tensor2im(mix[0])).astype(np.uint8)))
                # else:
                #     result_images.append(Image.fromarray(np.array(tensor2im(mixed_image[0])).astype(np.uint8)))
                result_images.append(Image.fromarray(
                    np.array(tensor2im(mixed_image[0])).astype(np.uint8)))
                finalInversion = mixed_image
            # #forward transition
            # for j in range(frames_pixel_interpolation):
            #     t2 = j/(frames_pixel_interpolation-1)
            #     t1 = 1- t2
            #     mix = finalInversion.cuda()* t1 + transformed_images[i+1].cuda()*t2
            #     result_images.append(Image.fromarray(np.array(tensor2im(mix[0])).astype(np.uint8)))

            # #backwards transition
            # for j in range(frames_pixel_interpolation):
            #     t2 = j/(frames_pixel_interpolation-1)
            #     t1 = 1- t2
            #     mix = transformed_images[i+1].cuda()* t1 + finalInversion.cuda()*t2
            #     result_images.append(Image.fromarray(np.array(tensor2im(mix[0])).astype(np.uint8)))

        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    print(result_images)
    result_images[0].save(fp='/content/drive/MyDrive/Github/facial-time-lapse-video/timelapse.gif', format='GIF', append_images=result_images,
                          save_all=True, duration=40, loop=0)

    # for index, output_image in enumerate(output_images):
    #    res_image = output_image
    #    res_image.save("output/result"+ str(index) +".jpg")
