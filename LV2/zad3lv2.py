import numpy as np
import matplotlib.pyplot as plt 

#a

image = plt.imread("road.jpg")
plt.imshow(image,alpha=0.3)
plt.show()

#b

splitImage = np.hsplit(image, 4)
plt.imshow (splitImage[2])
plt.show()

#c

rotatedImage = np.rot90(image, 3)
plt.imshow(rotatedImage)
plt.show()

#d

mirroredImage = np.fliplr(image)
plt.imshow(mirroredImage)
plt.show()