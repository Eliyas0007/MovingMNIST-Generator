import os
import cv2
import tqdm
import numpy
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from einops import rearrange


class Generator():
    def __init__(self, frame_len=20, step=4, direction='vertical', generation_path='.'):
        
        '''
        Direction Options
        "vertival"
        "horizontal"
        "diagonal"
        '''
        self.direction = direction 
        self.step = step
        self.frame_len = frame_len
        self.generation_path = generation_path

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))


    def generate(self):

        root_path = self.generation_path + f'/data_{self.direction}'
        os.mkdir(root_path)

        for item in tqdm.tqdm(range(len(self._dataset))):
    
            image, _ = self._dataset[item]
            image = image.cpu().detach().numpy()

            i = 0
            i_p = 0
            moving_down = True

            

            video_path = f'/video{item}'
            os.mkdir(root_path + video_path)

            while(i_p < self.frame_len):

                canvas = numpy.zeros((1, 64, 64))

                if i_p == 0:
                    top = self.step*i
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

                top = self.step*i
                bottom = top + 28

                canvas[:, top:bottom, 0:28] += image 


                canvas = rearrange(canvas, 'c h w -> h w c')
                image_path = root_path + video_path + f'/frame{i_p}.png'
                # cv2.imshow('asd', canvas)
                # cv2.waitKey(150)
                cv2.imwrite(image_path, canvas * 256)
                canvas = rearrange(canvas, 'h w c -> c h w')

                i_p += 1

