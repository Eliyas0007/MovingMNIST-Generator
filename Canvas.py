import sys
import cv2
import numpy
import random

from Digit import Digit

# random.seed(int(sys.argv[1]))

class Canvas:

    def __init__(self, size: tuple=(64, 64), num_digits=2, collision=True) -> None:
        
        self.height = size[0]
        self.width = size[1]

        self.digits = []
        self.processes = []
        self.collision = collision
        self.num_digits = num_digits

        self.canvas = numpy.zeros((num_digits, self.height, self.width))
        self.combined_canvas = numpy.zeros((self.height, self.width))
        self.motion_types = ['vertical', 'horizontal', 'diagonal']

        for _ in range(num_digits):
            self.digits.append(Digit())
        self.init_digit_positions()


    def place_digit_on_canvas(self, digit: Digit, digit_index):

        x, y = digit.coordinate[0], digit.coordinate[1]
        d_x, d_y, d_h, d_w = digit.digit_xyhw
        x += d_x
        y += d_y
        
        self.canvas[digit_index][:, :] = 0
        self.canvas[digit_index][y:y+d_h, x:x+d_w] += digit.image[d_y:d_y+d_h, d_x:d_x+d_w]


    def check_digit_ditection(self, digit: Digit, digit_index, step):

        margin = step
        collision_density = 5
        pixel_sum_x = 0
        pixel_sum_y = 0

        x, y = digit.coordinate[0], digit.coordinate[1]

        d_x, d_y, d_h, d_w = digit.digit_xyhw
        d_x += x
        d_y += y

        step = step * 2
        if digit.is_right:
            d_x_t = d_x + step
        else:
            d_x_t = d_x - step
        if digit.is_down:
            d_y_t = d_y + step
        else:
            d_y_t = d_y - step

        if self.collision:
            for i, c in enumerate(self.canvas):
                if i != digit_index:
                    pixel_sum_x += numpy.sum(c[d_y:d_y+d_h, d_x_t:d_x_t+d_w])
                    pixel_sum_y += numpy.sum(c[d_y_t:d_y_t+d_h, d_x:d_x+d_w])
        
        if (d_x_t + d_w > self.width - margin) or (d_x_t < margin) or (pixel_sum_x > collision_density):
            digit.change_direction_x()

        if (d_y_t + d_h > self.height - margin) or (d_y_t < margin) or (pixel_sum_y > collision_density):
            digit.change_direction_y()


    def move_digits(self, motion_type='random', step=2, rotation=False, zoom=False):

        motions = []
        if motion_type == 'random':
            for _ in range(self.num_digits):
                motions.append(random.sample(self.motion_types, 1)[0])
        else:
            motions = [motion_type]

        for i, digit in enumerate(self.digits):
            self.check_digit_ditection(digit, i, step)

        for i, digit in enumerate(self.digits):
            digit.move_by_direction(motions[i if motion_type == 'random' else 0], step)

        for i, digit in enumerate(self.digits):
            self.place_digit_on_canvas(digit, i)
            
        self.combined_canvas[:, :] = 0
        for canvas in self.canvas:
            self.combined_canvas += canvas


    def init_digit_positions(self):

        for i, digit in enumerate(self.digits):
            x = random.randint(0, self.canvas[i].shape[1]-digit.image.shape[1])
            # the reason why it is devided by 2 is make sure there are rooms for other digit
            y = random.randint(0, (self.canvas[i].shape[0] // 2)-digit.image.shape[0])
            h, w = digit.image.shape
            if self.collision:
                while True:
                    if numpy.sum(self.combined_canvas[y:y+h, x:x+w]) != 0:
                        x = random.randint(0, self.canvas[i].shape[1]-digit.image.shape[1])
                        y = random.randint(0, self.canvas[i].shape[0]-digit.image.shape[0])
                    else:
                        break
            digit.set_coordinate([x, y])
            self.place_digit_on_canvas(digit, i)

            for canvas in self.canvas:
                self.combined_canvas += canvas 
