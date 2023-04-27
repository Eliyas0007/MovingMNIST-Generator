# from generator import Generator
from Generator import Generator

if __name__ == '__main__':
    '''
        There are 5 main parameters 

        frame_len:
            the length of frame you want to generate
            default is 20

        step:
            how many pixeles you want to move for each time step
            default is 4

        direction: 
            how the digit will move, we provide 4 types of direction which is vertical,
                                                                              horizontal,
                                                                              diagonal,
                                                                              circular
            default is vertical

        num_digits:
            determine the numbers of appearing digits maximum 2 in vertical or horizontal direction,
                                                              3 in diagonal direction,
                                                              technically infinite numbers for this one but we recommend no more than 4
            default is 1

        canvas_size:
            it determines the size of frames
            default is 64
                                        
        generation_path:
            for saving generated dataset

            default is current directory which is '.'
    '''
    # directions = ['vertical']#, 'horizontal', 'circular']

    generator = Generator( 
                 frame_length=20,
                 step=2,
                 canvas_size=(64, 64),
                 num_digits=2,
                 collision=True,
                 direction='random',
                 rotate_digits=False,
                 zoom_digits=False,
                 num_of_videos=1,
                 generation_path='./train')

    # generator.show_example()
    generator.generate()
