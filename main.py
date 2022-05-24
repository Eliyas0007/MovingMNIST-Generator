import random
from generator import Generator

if __name__ == '__main__':
    '''
        There are 5 parameters 

        frame_len:
            the length of frame you want to generate
            default is 20

        step:
            how many pixeles you want to move for each time step
            default is 4

        direction: 
            how the digit will move, we provide 3 types of direction which is vertical,
                                                                              horizontal,
                                                                              diagonal,
                                                                              circular
            default is vertical

        num_digits:
            determine the numbers of appearing digits maximum 2 in vertical or horizontal direction,
                                                              3 in diagonal direction,
                                                              technically infinite numbers for this one but we recommend no more than 4
            default is 1
                                        
        generation_path:
            for saving generated dataset

            default is current directory which is '.'
    '''
    generator = Generator(frame_len=20, step=3, num_digits=1, direction='circular')
    generator.show_example(random.randrange(0, 10000))
    # generator.generate()
