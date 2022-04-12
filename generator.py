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
        directions = ['vertical', 'horizontal', 'diagonal']

        for i, d in enumerate(directions):
            if direction == d:
                break
            if direction != d and i == (len(directions) - 1):
                raise ValueError(f'Direction: [{direction}] is NOT supported moving direction!')
            
        self.direction = direction 
        self.step = step
        self.frame_len = frame_len
        self.generation_path = generation_path

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))
        self._is_forward = True
        self._iter = 0


    def _move_image(self, image):

        canvas = numpy.zeros((1, 64, 64))
        
        margin_f = self.step * self._iter
        margin_b = margin_f + 28

        if self._is_forward:
            self._iter += 1
            if margin_b > 60:
                self._is_forward = False
                self._iter -= 1
                return
        else:
            self._iter -= 1
            if margin_f < 1:
                self._is_forward = True
                self._iter += 1
                return


        if self.direction == 'vertical':
            canvas[:, margin_f:margin_b, 0:28] += image
        elif self.direction == 'horizontal':
            canvas[:, 0:28, margin_f:margin_b] += image
        elif self.direction == 'diagonal':
            canvas[:, margin_f:margin_b, margin_f:margin_b] += image

        return canvas
                   

    def generate(self):

        root_path = self.generation_path + f'/data_{self.direction}'
        os.mkdir(root_path)

        for item in tqdm.tqdm(range(len(self._dataset))):
    
            image, _ = self._dataset[item]
            image = image.cpu().detach().numpy()

            video_path = f'/video{item}'
            os.mkdir(root_path + video_path)

            self._iter = 0
            
            f = 0
            while f < self.frame_len:

                canvas = numpy.zeros((1, 64, 64))

                canvas = self._move_image(image)
                if canvas is None:
                    if f == 0:
                        continue
                    f -= 1
                    continue

                canvas = rearrange(canvas, 'c h w -> h w c')
                image_path = root_path + video_path + f'/frame{f}.png'
                cv2.imwrite(image_path, canvas * 256)
                canvas = rearrange(canvas, 'h w c -> c h w')
                f += 1


    def show_example(self, index: int):

        image, _ = self._dataset[index]
        image = image.cpu().detach().numpy()

        f = 0
        while f < self.frame_len:
            canvas = numpy.zeros((1, 64, 64))
            canvas = self._move_image(image)
            if canvas is None:
                f -= 1
                continue

            canvas = rearrange(canvas, 'c h w -> h w c')
            cv2.imshow(f'example {index}', canvas)
            cv2.waitKey(200)
            cv2.imwrite(f'./GeneratedExample/example{f}.png', canvas * 256)
            canvas = rearrange(canvas, 'h w c -> c h w')
            f += 1
