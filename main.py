from generator import Generator

if __name__ == '__main__':
    '''
        There are 3 parameters 

        frame_len:
            the length of frame you want to generate
            default is 20

        step:
            how many pixeles you want to move for each time step
            default is 4

        direction: 
            how the digit will move, we provide 3 types of direction which is vertical,
                                                                              horizontal,
                                                                              diagonal
            default is vertical
                                        
        generation_path:
            for saving generated dataset
    '''
    generator = Generator(frame_len=20, step=4, direction='vertical')
    generator.generate()