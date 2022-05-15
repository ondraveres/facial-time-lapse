from natsort import natsorted
from PIL import Image
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for index, filename in enumerate(natsorted(os.listdir(folder))):
        if(index % 4 != 0):
            continue
        img = np.asarray(Image.open(os.path.join(folder, filename)).convert('RGB').resize((200,200)))
        if img is not None:
            images.append(img)
    return images

import math
def gallery(array, ncols=math.ceil(27/4)):
    print(array.shape)
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

real= load_images_from_folder('data/dan-al')
pixels= load_images_from_folder('data/pixel')
psp0s= load_images_from_folder('data/psp0')
psp50s= load_images_from_folder('data/psp50')
restyle0s= load_images_from_folder('data/restyle0')
restyle50s= load_images_from_folder('data/restyle50')

everything = real+ pixels + psp0s + psp50s + restyle0s + restyle50s

result = Image.fromarray(gallery(np.array(everything)))
result.save('image_grid.png')