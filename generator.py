import cv2
import numpy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from einops import rearrange



mnist_dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))
frame_len = 20
step = 4
canvas = numpy.zeros((1, 64, 64))
image, _ = mnist_dataset[0]
image = image.cpu().detach().numpy()
moving_down = True

i = 0
i_p = 0

while(i_p < frame_len):
    print(i_p)
    canvas = numpy.zeros((1, 64, 64))
    
    if i_p == 0:
        top = step*i
        bottom = top + 28

    if moving_down:
        i += 1
        if bottom > 60:
            moving_down = False
            continue
    else:
        i -= 1
        if top < 1:
            moving_down = True
            continue

    top = step*i
    bottom = top + 28

    canvas[:, top:bottom, 0:28] += image 

    i_p += 1
    canvas = rearrange(canvas, 'c h w -> h w c')
    cv2.imshow('asd', canvas)
    cv2.waitKey(150)
    canvas = rearrange(canvas, 'h w c -> c h w')
    



