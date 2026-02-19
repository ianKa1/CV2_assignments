from PIL import Image
from harris import get_harris_corners, dist_SSD

image1 = Image.open('Assignment2/images/mac1.jpg')
image2 = Image.open('Assignment2/images/mac2.jpg')

h1, coords1 = get_harris_corners(image1)
h2, coords2 = get_harris_corners(image2)
print(coords1.shape, coords2.shape)

dist = dist_SSD(coords1, coords2)
print(dist)