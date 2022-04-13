import os
import cv2
import tqdm
import numpy
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from einops import rearrange


class Generator():

    def __init__(self, frame_len=20, step=4, direction='vertical', num_digits=2, generation_path='.'):
        
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
        self.num_digits = num_digits

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))
        self._is_forwards = []
        self._is_first_frame = True
        self._iters = []


    def _move_image(self, image, initial_position=(0, 0), iter_index=0):

        canvas = numpy.zeros((1, 64, 64))
 
        margin_f = self.step * self._iters[iter_index]
        margin_b = margin_f + 28

        if self._is_forwards[iter_index]:
            self._iters[iter_index] += 1
            if margin_b > 60:
                self._is_forwards[iter_index] = False
                self._iters[iter_index] -= 2
                return
        else:
            self._iters[iter_index] -= 1
            if margin_f < 1:
                self._is_forwards[iter_index] = True
                self._iters[iter_index] += 2
                return


        if self.direction == 'vertical':
            canvas[:, margin_f:margin_b, 0+initial_position[0]:28+initial_position[0]] += image

        elif self.direction == 'horizontal':
            canvas[:, 0+initial_position[0]:28+initial_position[0], margin_f:margin_b] += image

        elif self.direction == 'diagonal':
            canvas[:, margin_f:margin_b, margin_f:margin_b] += image

        return canvas
                   

    def generate(self):

        try:
            root_path = self.generation_path + f'/data_{self.direction}_{self.num_digits}_digits'
            os.mkdir(root_path)
        except FileExistsError:
            ...

        is_continue = False
        for index in tqdm.tqdm(range(len(self._dataset))):

            images = []

            for i in range(self.num_digits):

                if i >= len(self._dataset):
                    image, _ = self._dataset[index+i - len(self._dataset)]
                else:
                    image, _ = self._dataset[index+i]

                image = image.cpu().detach().numpy()
                images.append(image)
                self._iters.append(i * 10)
                self._is_forwards.append(True)
            
            try:
                video_path = f'/video{index}'
                os.mkdir(root_path + video_path)
            except FileExistsError:
                ...
            
            f = 0
            while f < self.frame_len:
                if f == 0:
                    self._is_first_frame = True
                elif f > 0 :
                    self._is_first_frame = False

                canvas = numpy.zeros((1, 64, 64))

                for i, image in enumerate(images):
                    y = i * 32
                    x = i * 35
                    new = self._move_image(image, (y, x), i)
                    

                    if new is None:
                        if f == 0:
                            continue
                        f -= 1
                        is_continue = True
                        continue
                    else:
                        canvas += new

                if is_continue:
                    is_continue = False
                    continue

                canvas = rearrange(canvas, 'c h w -> h w c')
                image_path = root_path + video_path + f'/frame{f}.png'
                cv2.imshow('asd', canvas)
                cv2.waitKey(200)
                cv2.imwrite(image_path, canvas * 256)
                canvas = rearrange(canvas, 'h w c -> c h w')
                f += 1
            break


    def show_example(self, index: int):

        image, _ = self._dataset[index]
        image = image.cpu().detach().numpy()

        f = 0
        while f < self.frame_len:
            canvas = numpy.zeros((1, 64, 64))
            canvas = self._move_image(image), 
            if canvas is None:
                f -= 1
                continue

            canvas = rearrange(canvas, 'c h w -> h w c')
            cv2.imshow(f'example {index}', canvas)
            cv2.waitKey(200)
            cv2.imwrite(f'./GeneratedExample/example{f}.png', canvas * 256)
            canvas = rearrange(canvas, 'h w c -> c h w')
            f += 1
