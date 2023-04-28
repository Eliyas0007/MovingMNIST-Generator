import os
import cv2
import tqdm
import glob
import numpy
import shutil
import random

from Canvas import Canvas

class Generator():

    def __init__(self, 
                 frame_length=20,
                 step=2,
                 canvas_size=(64, 64),
                 num_digits=2,
                 collision=True,
                 direction='random',
                 rotate_digits=False,
                 zoom_digits=False,
                 num_of_videos=1,
                 num_mask = 1,
                 generation_path='.') -> None:
        
        self.step = step
        self.num_mask = num_mask
        self.collision = collision
        self.direction = direction
        self.num_digits = num_digits
        self.frame_len = frame_length
        self.canvas_size = canvas_size
        self.zoom_digits = zoom_digits
        self.rotate_digits = rotate_digits
        self.num_of_videos = num_of_videos
        self.generation_path = generation_path
        
        # self.canvas = Canvas(canvas_size, num_digits, collision)

    def generate(self):

        try:
            root_path = self.generation_path + f'/data_{self.direction}_{self.num_digits}d_{self.canvas_size[0]}h_{self.canvas_size[1]}w_{self.frame_len}f_r{1 if self.rotate_digits else 0}_co{1 if self.collision else 0}'
            if os.path.exists(root_path):
                print('Deleting old files...')
                shutil.rmtree(root_path)
                print('Old files deleted!')
            os.mkdir(root_path)
        except FileExistsError:
            ...

        for index in tqdm.tqdm(range(self.num_of_videos)):
            
            try:
                video_path = f'/video{index}'
                os.mkdir(root_path + video_path)

                video_original_path = video_path + '/original'
                os.mkdir(root_path + video_original_path)

                sub_digit_paths = []
                for i in range(self.num_digits):
                    temp = video_path + f'/s{i}'
                    try:
                        os.mkdir(root_path + temp)
                    except FileExistsError or FileNotFoundError:
                        ...
                    sub_digit_paths.append(temp)
    
            except FileExistsError or FileNotFoundError:
                ...
            
            mask_indeces = random.sample([n for n in range(self.num_digits)], self.num_mask)
            canvas = Canvas(self.canvas_size, self.num_digits, self.collision)
            for f in range(self.frame_len):
                canvas.move_digits(self.direction, self.step) 

                for mask_index in mask_indeces:
                    canvas.canvas[mask_index][:, :] = 0

                canvas.update_combinde_canvas()
                frame_o = canvas.combined_canvas
                image_path_o = root_path + video_original_path + f'/frame_o_{f}.png'
                cv2.imwrite(image_path_o, frame_o * 255)
                canvas_without_padding = canvas.get_canvas_without_padding()
                for i in range(self.num_digits):
                    frame_i = canvas_without_padding[i]
                    image_path_i = root_path + sub_digit_paths[i] + f'/frame_s{i}_{f}.png'
                    cv2.imwrite(image_path_i, frame_i * 255)

    def show_example(self):

        old_images_paths = glob.glob('./GeneratedExample/*.png')
        if len(old_images_paths) > 0:
            for path in old_images_paths:
                try:
                    os.remove(path)
                except:
                    print("Error while deleting file : ", path)

        mask_indeces = random.sample([n for n in range(self.num_digits)], self.num_mask)
        canvas = Canvas(self.canvas_size, self.num_digits, self.collision)
        clip = []
        for f in range(self.frame_len):
            canvas.move_digits(self.direction, self.step)
            
            for mask_index in mask_indeces:
                canvas.canvas[mask_index][:, :] = 0         
            canvas.update_combinde_canvas()
            frame = canvas.combined_canvas
            clip.append(frame)
            canvas_without_padding = canvas.get_canvas_without_padding()
            for i in range(self.num_digits):           
                frame = numpy.concatenate((frame, canvas_without_padding[i]), axis=1)

            # cv2.imshow(f'example', frame)
            # cv2.waitKey(100)
            # print(frame.shape)
            try: os.mkdir('./GeneratedExample')
            except FileExistsError: pass
            cv2.imwrite(f'./GeneratedExample/example{f}.png', frame * 255)
    