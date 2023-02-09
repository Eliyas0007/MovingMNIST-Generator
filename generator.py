import os
import cv2
import tqdm
import math
import glob
import numpy
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from einops import rearrange


class Generator():

    def __init__(self, 
                frame_len = 20,
                step = 1,
                direction = 'vertical',
                acceleration = 1,
                num_digits = 2,
                zoom = True,
                canvas_size=512,
                generation_path = '.'):
        
        '''
        Direction Options
        "vertival"
        "horizontal"
        "diagonal"
        '''
        directions = ['vertical', 'horizontal', 'diagonal', 'circular']

        for i, d in enumerate(directions):
            if direction == d:
                break
            if direction != d and i == (len(directions) - 1):
                raise ValueError(f'Direction: [{direction}] is NOT supported moving direction!')
            
        self.step = step
        self.direction = direction
        self.acceleration = acceleration
        self.frame_len = frame_len
        self.num_digits = num_digits
        self.zoom = zoom
        self.canvas_size = canvas_size
        self.generation_path = generation_path

        self._iters = []
        self._is_forwards = []
        self._zooms = []
        self._zoom_directions = []

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))


    def _move_image(self, image, initial_position=(0, 0), iter_index=0, spin_direction=0):

        canvas = numpy.zeros((1, self.canvas_size, self.canvas_size))
        _, h, w = image.shape
        
        if self.zoom:
            if self._zoom_directions[iter_index]:
                if self._zooms[iter_index] < (h/2) + 2:
                    self._zoom_directions[iter_index] = False
                self._zooms[iter_index] -= 2
            else:
                if self._zooms[iter_index] > 26:
                    self._zoom_directions[iter_index] = True
                self._zooms[iter_index] += 2

            image = cv2.resize(rearrange(image, 'c h w -> (c h) w'), (self._zooms[iter_index], self._zooms[iter_index]))
            image = rearrange(image, '(c h) w -> c h w', c=1)
        _, h, w = image.shape
                

        if self.direction == 'circular':
            radius = (self.canvas_size/2) - 14
            angle = (20/360)
            origin = self.canvas_size/2
            x = radius * math.sin(self._iters[iter_index] * math.pi * angle) + origin
            y = radius * math.cos(self._iters[iter_index] * math.pi * angle)+ origin

            # controlling the clockwise and anti-clockwise 
            if spin_direction == 0:
                canvas[:, int(x)-int(h/2):int(x)+int(w/2), int(y)-int(h/2):int(y)+int(w/2)] += image
            else:
                canvas[:, int(y)-int(h/2):int(y)+int(w/2), int(x)-int(h/2):int(x)+int(w/2)] += image

            self._iters[iter_index] += 1

            return canvas

        else:
            
            margin_f = self.step * self._iters[iter_index]
            margin_b = margin_f + h
            

            if self._is_forwards[iter_index]:
                self._iters[iter_index] += 1
                if margin_b >= self.canvas_size - 4:
                    self._is_forwards[iter_index] = False
                    self._iters[iter_index] -= 2
                    return
            else:
                self._iters[iter_index] -= 1
                if margin_f < 0:
                    self._is_forwards[iter_index] = True
                    self._iters[iter_index] += 2
                    return

            if self.direction == 'vertical':
                canvas[:, margin_f:margin_b, 0+initial_position[0]:h+initial_position[0]] += image

            elif self.direction == 'horizontal':
                canvas[:, 0+initial_position[0]:h+initial_position[0], margin_f:margin_b] += image

            elif self.direction == 'diagonal':
                canvas[:, margin_f:margin_b, margin_f:margin_b] += image

            self.step += self.acceleration

            return canvas
                   

    def generate(self):

        try:
            root_path = self.generation_path + f'/data_{self.direction}_{self.num_digits}digits_{self.canvas_size}size_{self.acceleration}ac'
            os.mkdir(root_path)
        except FileExistsError:
            ...

        for index in tqdm.tqdm(range(len(self._dataset))):
            
            try:
                video_path = f'/video{index}'
                os.mkdir(root_path + video_path)
            except FileExistsError:
                ...
            
            self._iters = []
            self.step = 1
            video, separated = self._make_video(index)
            
            for f, frame in enumerate(video):
                frame_o = rearrange(frame, 'c h w -> h w c')
                frame_s1 = rearrange(separated[0][f], 'c h w -> h w c')
                frame_s2 = rearrange(separated[1][f], 'c h w -> h w c')
                image_path_o = root_path + video_path + f'/frame_o_{f}.png'
                image_path_s1 = root_path + video_path + f'/frame_s1_{f}.png'
                image_path_s2 = root_path + video_path + f'/frame_s2_{f}.png'
                cv2.imwrite(image_path_o, frame_o * 256)
                cv2.imwrite(image_path_s1, frame_s1 * 256)
                cv2.imwrite(image_path_s2, frame_s2 * 256)
    
    
    def _make_video(self, index):

        video = []
        images = []

        separated = [[], []]

        for i in range(self.num_digits):

            if index >= (len(self._dataset)-self.num_digits):
                image, _ = self._dataset[index+i - len(self._dataset)]
            else:
                image, _ = self._dataset[index+i]

            image = image.cpu().detach().numpy()
            images.append(image)
            if self.direction == 'circular':
                self._iters.append(i * random.randint(1, 100) + random.randint(1, 100))
                self._zooms.append(random.randint(7, 14) * 2)
            else:
                self._iters.append(i * 10)
                self._zooms.append(random.randint(14, 28))

            self._is_forwards.append(True)
            self._zoom_directions.append(True)
            
        spin_direction = random.randint(0, 1)

        done = False
        while done is not True:         
            for i, image in enumerate(images):
                y = i * 32
                x = i * 32
                new = self._move_image(image, (y, x), i, spin_direction)
                
                if new is None:
                    ...
                else:
                    if len(separated[i]) < self.frame_len:
                        separated[i].append(new)
            
            for k, sep_seq in enumerate(separated):
                if len(sep_seq) < self.frame_len:
                    done = False
                    break
                done = True

            if done:
                for f in range(self.frame_len):
                    canvas = numpy.zeros((1, self.canvas_size, self.canvas_size))
                    for n in range(self.num_digits):
                        print(n, f)
                        canvas += separated[n][f]
                    video.append(canvas)        

        print(len(video), len(separated[0]), len(separated[1]))
        return video, separated


    def show_example(self, index: int):

        old_images_paths = glob.glob('./GeneratedExample/*.png')
        if len(old_images_paths) > 0:
            for path in old_images_paths:
                try:
                    os.remove(path)
                except:
                    print("Error while deleting file : ", path)


        video, separated = self._make_video(index)

        for f, frame in enumerate(video):
            frame = rearrange(frame, 'c h w -> h w c')
            frame1 = rearrange(separated[0][f], 'c h w -> h w c')
            frame2 = rearrange(separated[1][f], 'c h w -> h w c')
            cv2.imshow(f'example {index}', numpy.concatenate((frame, frame1, frame2), axis=1))

            cv2.waitKey(150)
            cv2.imwrite(f'./GeneratedExample/example{f}.png', frame * 256)
