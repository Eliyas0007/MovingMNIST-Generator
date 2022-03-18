import os
import cv2
import tqdm
import numpy
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from einops import rearrange



mnist_dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))


frame_len = 20
step = 4

for item in tqdm.tqdm(range(len(mnist_dataset))):
    
    image, _ = mnist_dataset[item]
    image = image.cpu().detach().numpy()
    i = 0
    i_p = 0
    moving_down = True
    
    root_path = f'./GeneratedData/video{item}'
    os.mkdir(root_path)
    
    while(i_p < frame_len):
        
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

        
        canvas = rearrange(canvas, 'c h w -> h w c')
        image_path = root_path + f'/frame{i_p}.png'
        # cv2.imshow('asd', canvas)
        # cv2.waitKey(150)
        cv2.imwrite(image_path, canvas)
        canvas = rearrange(canvas, 'h w c -> c h w')

        i_p += 1




