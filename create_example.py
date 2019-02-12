# Import the essential libraries
import os
import csv
import cv2
import matplotlib.image as npimg
import numpy as np
import matplotlib.pyplot as plt

name = 'examples/center_sample.jpg'
center_image = npimg.imread(name)
# center_flipped = np.fliplr(center_image)
((50,20), (0,0))
npimg.imsave('examples/center_cropped.jpg', center_image[0:50, 0:20])
