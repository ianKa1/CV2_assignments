import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.ndimage import shift
from skimage.transform import rescale

BORDER_SIZE = 30


# try displacement [-search_range, search_range] for each axis
def simple_align(channel, reference, search_range=[-15, 15]):
    H, W = reference.shape
    best_shift = (0, 0)
    minimal_mean_distance = np.inf
    for dx in range(search_range[0], search_range[1] + 1):
        for dy in range(search_range[0], search_range[1] + 1):
            # channel shifted by (dx, dy) - reference
            rx0 = max(0, dx)
            ry0 = max(0, dy)
            rx1 = min(H, H + dx)
            ry1 = min(W, W + dy)

            if rx1 <= rx0 or ry1 <= ry0:
                continue

            cx0 = max(0, -dx)
            cy0 = max(0, -dy)
            cx1 = cx0 + (rx1 - rx0)
            cy1 = cy0 + (ry1 - ry0)

            mean_distance = np.mean((channel[cx0:cx1, cy0:cy1] - reference[rx0:rx1, ry0:ry1]) ** 2)
            if mean_distance < minimal_mean_distance:
                minimal_mean_distance = mean_distance
                best_shift = (dx, dy)
    
    return best_shift, shift(channel, shift=(best_shift[0], best_shift[1]), order=1, mode='nearest', cval=0.0)

# scale = 2
def pyramid_align(channel, reference, search_range=15, border_size=BORDER_SIZE, num_levels=3):
    current_shift = (0, 0)
    for level in range(num_levels - 1, -1, -1):
        scale = 0.5 ** level
        if scale < 1.0:
            channel_downsampled = rescale(channel, scale, anti_aliasing=True, preserve_range=True)
            reference_downsampled = rescale(reference, scale, anti_aliasing=True, preserve_range=True)
        else:
            channel_downsampled = channel
            reference_downsampled = reference
        
        



simple_align_image_files = ['coms4732_hw1_data/cathedral.jpg', 'coms4732_hw1_data/monastery.jpg', 'coms4732_hw1_data/tobolsk.jpg']

for simple_align_image_file in simple_align_image_files:
    print(f"Processing {simple_align_image_file} with simple alignment...\n Align to blue channel...")
    image = skio.imread(simple_align_image_file)
    image = sk.img_as_float(image)

    height = np.floor(image.shape[0] / 3.0).astype(int)

    b = image[:height]
    g = image[height: 2*height]
    r = image[2*height: 3*height]
    # remove borders by constant size
    g = g[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]
    r = r[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]
    b = b[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]

    g_shift, g_aligned_image = simple_align(g, b)
    r_shift, r_aligned_image = simple_align(r, b)

    print(f"Green Channel Shift: {g_shift}, Red Channel Shift: {r_shift}")
    im_out = np.dstack([r_aligned_image, g_aligned_image, b])
    im_out = np.clip(im_out, 0, 1)
    im_out_uint8 = (im_out * 255).astype(np.uint8)

    output_name = simple_align_image_file.split('/')[-1].split('.')[0] + '_simple_aligned.jpg'
    
    skio.imsave(output_name, im_out_uint8)
    print(f"Image saved successfully as {output_name}")