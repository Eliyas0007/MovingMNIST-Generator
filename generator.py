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
                direction = 'circular',
                acceleration = 1,
                num_digits = 2,
                zoom = True,
                canvas_size=512,
                num_of_videos=10000,
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
        self.num_of_videos = num_of_videos
        self.generation_path = generation_path

        self._iters = []
        self._is_forwards = []
        self._zooms = []
        self._zoom_directions = []
        self._initial_positions = []

        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))


    def _move_image(self, image, iter_index=0, spin_direction=0):

        # print(initial_position)

        canvas = numpy.zeros((1, self.canvas_size, self.canvas_size))
        _, h, w = image.shape
        
        if self.zoom:
            if self._zoom_directions[iter_index]:
                if self._zooms[iter_index] < (h/2) + 2:
                    self._zoom_directions[iter_index] = False
                self._zooms[iter_index] -= 2
            else:
                if self._zooms[iter_index] > 24:
                    self._zoom_directions[iter_index] = True
                self._zooms[iter_index] += 2

            image = cv2.resize(rearrange(image, 'c h w -> (c h) w'), (self._zooms[iter_index], self._zooms[iter_index]))
            image = rearrange(image, '(c h) w -> c h w', c=1)
        _, h, w = image.shape


        if self.direction == 'circular':
            radius = (self.canvas_size // 2) - 14
            angle = (20 / 360)
            origin = self.canvas_size // 2
            x = radius * math.sin(self._iters[iter_index] * math.pi * angle) + origin
            y = radius * math.cos(self._iters[iter_index] * math.pi * angle) + origin
            # print(self._iters[iter_index], math.pi, angle, origin)

            # controlling the clockwise and anti-clockwise movement
            if spin_direction == 0:
                canvas[:, int(x)-int(h/2):int(x)+int(w/2), int(y)-int(h/2):int(y)+int(w/2)] += image
            else:
                canvas[:, int(y)-int(h/2):int(y)+int(w/2), int(x)-int(h/2):int(x)+int(w/2)] += image

            self._iters[iter_index] += 1

            return canvas

        else:
            margin_f = self.step * self._iters[iter_index] + self._initial_positions[iter_index]
            margin_b = margin_f + h

            # print(margin_f, margin_b, self._iters[iter_index], self.step, self.canvas_size)
            

            if self._is_forwards[iter_index]:
                self._iters[iter_index] += 1
                if margin_b >= self.canvas_size:
                    self._is_forwards[iter_index] = False
                    self._iters[iter_index] -= 1
                    # return
            else:
                self._iters[iter_index] -= 1
                if margin_f < 0:
                    self._is_forwards[iter_index] = True
                    self._iters[iter_index] += 1
                    # return

            if margin_b <= 64 and margin_f > 0:
                y = self._initial_positions[iter_index]
                if self.direction == 'vertical':
                    canvas[:, margin_f:margin_b, y:h+y] += image

                elif self.direction == 'horizontal':
                    canvas[:, y:h+y, margin_f:margin_b] += image

                elif self.direction == 'diagonal':
                    canvas[:, margin_f:margin_b, margin_f:margin_b] += image
            else: return

            self.step += self.acceleration

            return canvas
                   

    def generate(self):

        try:
            root_path = self.generation_path + f'/data_Z_{self.direction}_{self.num_digits}digits_{self.canvas_size}size_{self.acceleration}ac'
            os.mkdir(root_path)
        except FileExistsError:
            ...

        for index in tqdm.tqdm(range(self.num_of_videos)):
            
            try:
                video_path = f'/video{index}'
                os.mkdir(root_path + video_path)
            except FileExistsError:
                ...
            
            self._iters = []
            video, separated = self._make_video()
            
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
    
    
    def _make_video(self):

        video = []
        images = []

        separated = [[], []]

        self._zooms = []
        self._iters = []
        self._is_forwards = []
        self._zoom_directions = []
        self._initial_positions = []

        for i in range(self.num_digits):

            image, _ = self._dataset[random.randint(0, len(self._dataset)-1)]

            image = image.cpu().detach().numpy()
            images.append(image)

            if self.direction == 'circular':
                self._iters.append(i * random.randint(1, 100) + random.randint(1, 100))
                self._zooms.append(random.randint(7, 14) * 2)
            else:
                self._iters.append(i * random.randint(0, 31))
                self._zooms.append(random.randint(14, 28))
                self._initial_positions.append(random.randint(0, 64-28))

            self._is_forwards.append(random.getrandbits(1))
            self._zoom_directions.append(random.getrandbits(1))
            
        spin_direction = random.randint(0, 1)

        done = False
        while done is not True:
            for i, image in enumerate(images):
                new = self._move_image(image, i, spin_direction)
                
                if new is None:
                    ...
                else:
                    if len(separated[i]) < self.frame_len:
                        separated[i].append(new)
            
            for _, sep_seq in enumerate(separated):
                if len(sep_seq) < self.frame_len:
                    done = False
                    break
                done = True

            if done:
                for f in range(self.frame_len):
                    canvas = numpy.zeros((1, self.canvas_size, self.canvas_size))
                    for n in range(self.num_digits):
                        canvas += separated[n][f]
                    video.append(canvas)        

        return video, separated


    def show_example(self):

        old_images_paths = glob.glob('./GeneratedExample/*.png')
        if len(old_images_paths) > 0:
            for path in old_images_paths:
                try:
                    os.remove(path)
                except:
                    print("Error while deleting file : ", path)


        video, separated = self._make_video()

        for f, frame in enumerate(video):
            frame = rearrange(frame, 'c h w -> h w c')
            frame1 = rearrange(separated[0][f], 'c h w -> h w c')
            frame2 = rearrange(separated[1][f], 'c h w -> h w c')
            # cv2.imshow(f'example {index}', numpy.concatenate((frame, frame1, frame2), axis=1))

            # cv2.waitKey(150)
            try: os.mkdir('./GeneratedExample')
            except FileExistsError: pass
            cv2.imwrite(f'./GeneratedExample/example{f}.png', frame * 256)
