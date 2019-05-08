
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import random
from torchvision.transforms import Normalize
import cv2


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        # print(return_images)
        return_images = torch.cat(return_images, 0)
        return return_images

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unnormalize(y, mean, std):
    """

    :param y: input image tensor, (Batch, Channels, Weight, Height)
    :param mean: mean, length of mean is 1 or 3
    :param std:  std, length of std is 1 or 3
    :return: unnormalize output image tensor. range:[0, 255]

    """

    if not isinstance(y, torch.Tensor):
        raise TypeError('input is not a tensor.')
    elif not y.dim() == 4:
        raise TypeError('input tensor is not 4 dimension.')
    else:
        x = y.new(*y.size())

    if x.size(1) == 3 and len(mean) == 3 and len(std) == 3:
        x[:, 0, :, :] = (y[:, 0, :, :] * std[0] + mean[0]) * 255
        x[:, 1, :, :] = (y[:, 1, :, :] * std[1] + mean[1]) * 255
        x[:, 2, :, :] = (y[:, 2, :, :] * std[2] + mean[2]) * 255
    elif x.size(1) == 1:
        x[:, 0, :, :] = (y[:, 0, :, :] * std[0] + mean[0]) * 255
    else:
        raise TypeError('input tensor and mean and std do not have the same channels.')

    return x.clamp(0, 255)


def tensor2im(input_image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
        image_tensor = unnormalize(image_tensor, mean, std)
    else:
        return input_image

    image_numpy = image_tensor[0].cpu().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = np.squeeze(image_numpy, 2)
        image_pil = Image.fromarray(image_numpy, mode='L')
    elif image_numpy.shape[2] == 3:
        image_pil = Image.fromarray(image_numpy, mode='RGB')
    else:
        raise TypeError('image_pil is not 3 channel or 1 channel')
    image_pil.save(image_path)


def cv2PIL(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img)

    return pil_img

def PIL2cv(pil_img):

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return cv_img

if __name__ == '__main__':
    a = torch.zeros(1, 5, 5).float()
    # print(a)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    norm = Normalize(mean, std)
    k = norm(a).view(1, 1, 5, 5)
    print(k)
    q = unnormalize(k, mean, std)
    print(q)