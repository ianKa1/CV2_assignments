from PIL import Image, ImageDraw
from harris import get_harris_corners, dist_SSD
import numpy as np
from scipy.ndimage import maximum_filter

def convert_to_grayscale(image):
    return image.convert('L')

def visualize_corners(image, coords, output_path, radius=1, color='red'):
    """
    Visualize Harris corners on an image.

    Args:
        image: PIL Image object
        coords: numpy array of corner coordinates in shape (2, n) where coords[0] = ys, coords[1] = xs
        output_path: path to save the visualization
        radius: radius of corner markers (default: 3)
        color: color of corner markers (default: 'red')
    """
    image_with_corners = image.copy()
    draw = ImageDraw.Draw(image_with_corners)
    # coords is (2, n) where coords[0] = y coords, coords[1] = x coords
    for i in range(coords.shape[1]):
        y, x = coords[0, i], coords[1, i]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)
    image_with_corners.save(output_path)
    return image_with_corners

def nms_simple(h, window_size=25):
    h_max = maximum_filter(h, size=window_size)
    keep = h == h_max

    coords = np.vstack(np.nonzero(keep))
    ys = coords[0]
    xs = coords[1]
    mask = h[ys, xs] != 0

    coords = coords[:, mask]

    ys = coords[0]
    xs = coords[1]
    values = h[ys, xs]
    threshold_h = 0.02 * values.max()
    mask = h[ys, xs] > threshold_h
    coords = coords[:, mask]
    return coords

image1 = Image.open('images/mac1.jpg')
image2 = Image.open('images/mac2.jpg')
image1_gray = convert_to_grayscale(image1)
image1_gray = np.array(image1_gray)
image2_gray = convert_to_grayscale(image2)  
image2_gray = np.array(image2_gray)

# Step 1: Get Harris corners
h1, coords1 = get_harris_corners(image1_gray)
h2, coords2 = get_harris_corners(image2_gray)
print(coords1.shape, coords2.shape)

# Visualize Harris corners on original images
visualize_corners(image1, coords1, 'mac1_corners_STEP1.jpg', 1)
visualize_corners(image2, coords2, 'mac2_corners_STEP1.jpg', 1)

# Step 2: Non-maximum suppression
coords_nms1 = nms_simple(h1)
coords_nms2 = nms_simple(h2)
print(coords_nms1.shape, coords_nms2.shape)

# Visualize NMS corners on original images
visualize_corners(image1, coords_nms1, 'mac1_nms_STEP2.jpg', 3)
visualize_corners(image2, coords_nms2, 'mac2_nms_STEP2.jpg', 3)


