import cv2
import math
import numpy
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Digit:

    def __init__(self) -> None:
        
        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))
        temp_image = self._dataset[random.randint(0, len(self._dataset)-1)][0] * 1
        self.image = numpy.array(temp_image.squeeze(0).detach().numpy())
        # self.image = numpy.where(self.image > numpy.mean(self.image), 1.0, 0.0)
        self.coordinate = [0, 0]
        self.digit_xyhw = [0, 0, 0, 0]

        
        # self.real_shape = 
        self.rotation_angle = 0
        self.moving_angle_per_step = 10
        self.zoomed_image = None
        self.is_down = True if random.randint(0, 1) else False
        self.is_right = True if random.randint(0, 1) else False
        self.is_clock_wise = True if random.randint(0, 1) else False

        self.init_digit_status()

    def init_digit_status(self):
        self._get_real_shape()
    
    def _get_real_shape(self):
        # find the indices of all non-zero elements in the image
        binary = numpy.where(self.image > numpy.mean(self.image), 1.0, 0.0)
        nonzero_indices = numpy.argwhere(binary)

        min_row, min_col = numpy.min(nonzero_indices, axis=0)
        max_row, max_col = numpy.max(nonzero_indices, axis=0)

        h = max_row - min_row + 1
        w = max_col - min_col + 1
        y = min_row
        x = min_col

        self.digit_xyhw = [x, y, h, w]

    def set_coordinate(self, coordinate: list):
        self.coordinate = coordinate

    def rotate_by_angle(self, angle):
        
        image = self.image
        self.rotation_angle = angle
        if self.zoomed_image is not None:
            image = self.zoomed_image
        (h, w) = image.shape
        center = (h // 2, w //2)
        M = cv2.getRotationMatrix2D(center, 0 + angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (h, w))
        return rotated
    
    def change_direction_x(self):
        if self.is_right:
            self.is_right = False
        else:
            self.is_right = True

    def change_direction_y(self):
        if self.is_down:
            self.is_down = False
        else:
            self.is_down = True

    def update_coordinate(self, x, y):
        self.coordinate[0] = x
        self.coordinate[1] = y


    def move_by_direction(self, direction, step):
        if direction == 'horizontal':
            if self.is_right:
                self.coordinate[0] += step
            else:
                self.coordinate[0] -= step
        elif direction == 'vertical':
            if self.is_down:
                self.coordinate[1] += step
            else:
                self.coordinate[1] -= step
        elif direction == 'diagonal':
            if self.is_right:
                self.coordinate[0] += step
            else:
                self.coordinate[0] -= step
            if self.is_down:
                self.coordinate[1] += step
            else:
                self.coordinate[1] -= step 
        elif direction == 'circular':
            raise NotImplementedError
            # radius = 32
            # if self.is_clock_wise:
            #     x = radius * math.sin(self.coordinate[0] * math.pi * self.moving_angle_per_step) + origin
            #     y = radius * math.cos(self.coordinate[1] * math.pi * self.moving_angle_per_step) + origin
            #     self.coordinate[0] = 
            #     self.coordinate[1]
            # else:
            #     self.coordinate[0]
            #     self.coordinate[1]
        else:
            raise NotImplementedError

    def zoom_by_pixel_size(self, pixel_size: int):
        image = self.image
        if self.rotation_angle != 0:
            image = self.rotate_by_angle(self.rotation_angle)
        pixel_size = int(7 * numpy.sin(pixel_size/4) + 21)
        zoomed = cv2.resize(image, (pixel_size, pixel_size))
        return zoomed
