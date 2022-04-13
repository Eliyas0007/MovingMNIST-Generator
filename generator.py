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
            
        self.step = step
        self.direction = direction 
        self.frame_len = frame_len
        self.num_digits = num_digits
        self.generation_path = generation_path

        self._iters = []
        self._is_forwards = []
        self._is_first_frame = True

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))


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

        for index in tqdm.tqdm(range(len(self._dataset))):
            
            try:
                video_path = f'/video{index}'
                os.mkdir(root_path + video_path)
            except FileExistsError:
                ...
            
            video = self.make_video(index)
            
            for f, frame in enumerate(video):
                frame = rearrange(frame, 'c h w -> h w c')
                image_path = root_path + video_path + f'/frame{f}.png'
                cv2.imwrite(image_path, frame * 256)
    
    
    def make_video(self, index):

        images = []
        is_continue = False
        video = []

        for i in range(self.num_digits):

            if index >= (len(self._dataset)-1-self.num_digits):
                image, _ = self._dataset[index+i - len(self._dataset)]
            else:
                image, _ = self._dataset[index+i]

            image = image.cpu().detach().numpy()
            images.append(image)
            self._iters.append(i * 10)
            self._is_forwards.append(True)
    
            
        f = 0
        b = 0
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

            if b == 20:
                break

            video.append(canvas)
            f += 1
            b += 1

        return video


    def show_example(self, index: int):

        video = self.make_video(index)

        for f, frame in enumerate(video):
            frame = rearrange(frame, 'c h w -> h w c')
            cv2.imshow(f'example {index}', frame)
            cv2.waitKey(200)
            cv2.imwrite(f'./GeneratedExample/example{f}.png', frame * 256)
