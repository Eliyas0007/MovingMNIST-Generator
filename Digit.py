import cv2
import numpy
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

random.seed(1)

class Digit:

    def __init__(self) -> None:
        
        self._dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]))
        temp_image = self._dataset[random.randint(0, len(self._dataset)-1)][0]
        self.image = numpy.array(temp_image.squeeze(0).detach().numpy())
        self.coordinate = (0, 0)

    def set_coordinate(self, coordinate: tuple):
        self.coordinate = coordinate

    def set_image(self, image):
        self.image = image

    def rotate_by_angle(self, angle):

        (h, w) = self.image.shape
        center = (h // 2, w //2)
        M = cv2.getRotationMatrix2D(center, 0 + angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (h, w))
        self.set_image(rotated)

    def zoom_by_pixel_size(self, pixel_size: int):
        h, w = self.image.shape
        self.set_image(cv2.resize(self.image, (h + pixel_size, w + pixel_size)))


if __name__ == '__main__':
    digit = Digit()
    digit.rotate_by_angle(-30)
    cv2.imshow('exmple', digit.image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print(digit.image.shape)