import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.ndimage import shift
from skimage.transform import rescale

BORDER_SIZE = 30


# try displacement [-search_range, search_range] for each axis
def simple_align(channel, reference, x_search_range=(-15, 15), y_search_range=(-15, 15)):
    H, W = reference.shape
    best_shift = (0, 0)
    minimal_mean_distance = np.inf
    for dx in range(x_search_range[0], x_search_range[1] + 1):
        for dy in range(y_search_range[0], y_search_range[1] + 1):
            # channel shifted by (dx, dy) - reference
            rx0 = max(0, dx)
            ry0 = max(0, dy)
            rx1 = min(H, H + dx)
            ry1 = min(W, W + dy)

            if rx1 <= rx0 or ry1 <= ry0:
                continue
            if rx1 - rx0 <= 0.9 * channel.shape[0] or ry1 - ry0 <= 0.8 * channel.shape[1]:
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

def gaussian_kernel_1d(sigma=1.0):
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def gaussian_1d_conv_reflect(image, k, axis):
    pad_size = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (pad_size, pad_size)
    padded_image = np.pad(image, pad, mode = "reflect")

    conv_image = np.zeros_like(image, dtype = np.float32)
    if axis == 0:
        for i, w in enumerate(k):
            conv_image += w * padded_image[i:i+image.shape[0], :]
    else:
        for i, w in enumerate(k):
            conv_image += w * padded_image[:, i:i+image.shape[1]]
    
    return conv_image

def gaussian_blur(image, sigma=1.0):
    k = gaussian_kernel_1d(sigma)
    blur_image = gaussian_1d_conv_reflect(image, k, axis = 0)
    blur_image = gaussian_1d_conv_reflect(blur_image, k, axis = 0)

    return blur_image

def build_pyramid(image, num_levels, sigma=1.0):
    image_pyramid = [image]
    cur_image = image

    for level in range(num_levels - 1):
        blur_image = gaussian_blur(cur_image, sigma=sigma)
        cur_image = blur_image[::2, ::2]
        image_pyramid.append(cur_image)

    return image_pyramid

# scale = 2
def pyramid_align(channel, reference, local_shift_range=3, num_levels=7):
    channel_pyramid = build_pyramid(channel, num_levels)
    reference_pyramid = build_pyramid(reference, num_levels)
    current_shift = (0, 0)
    shifted_image = []

    coarse_range = min(20, int(0.05 * min(channel.shape[0], channel.shape[1])))
    print(f'coarse range {coarse_range} size {channel.shape}')
    
    for level in range(num_levels - 1, -1, -1):
        scale = 0.5 ** level
        current_shift = (current_shift[0] * 2, current_shift[1]*2)

        channel_downsampled = channel_pyramid[level]
        reference_downsampled = reference_pyramid[level]
        
        x_shift_range = (0, 0)
        y_shift_range = (0, 0)
        if level == num_levels - 1:
            x_shift_range = (-int(coarse_range * scale), int(coarse_range * scale))
            y_shift_range = x_shift_range
        else:
            x_shift_range = (current_shift[0] - local_shift_range, current_shift[0] + local_shift_range)
            y_shift_range = (current_shift[1] - local_shift_range, current_shift[1] + local_shift_range)
            
        best_shift, shifted_image = simple_align(channel_downsampled, reference_downsampled, x_shift_range, y_shift_range)

        current_shift = best_shift

    return current_shift, shifted_image

def process_image(input_path, output_path, method):
    print(f"Processing {input_path} with {method}... Align to blue channel...")
    image = skio.imread(input_path)
    image = sk.img_as_float(image)

    height = np.floor(image.shape[0] / 3.0).astype(int)

    b = image[:height]
    g = image[height: 2*height]
    r = image[2*height: 3*height]
    # remove borders by constant size
    g = g[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]
    r = r[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]
    b = b[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE]
    if method == 'pyramid_alignment':
        g_shift, g_aligned_image = pyramid_align(g, b)
        r_shift, r_aligned_image = pyramid_align(r, b)
    else:
        if method == 'simple_alignment':
            g_shift, g_aligned_image = simple_align(g, b)
            r_shift, r_aligned_image = simple_align(r, b)

    print(f"Green Channel Shift: {g_shift}, Red Channel Shift: {r_shift}")
    im_out = np.dstack([r_aligned_image, g_aligned_image, b])
    im_out = np.clip(im_out, 0, 1)
    im_out_uint8 = (im_out * 255).astype(np.uint8)
    
    skio.imsave(output_path, im_out_uint8)
    print(f"Image saved successfully as {output_path}")
    

# simple_align_image_files = ['coms4732_hw1_data/cathedral.jpg', 'coms4732_hw1_data/monastery.jpg', 'coms4732_hw1_data/tobolsk.jpg']
# # pyramid_align_image_files = ['coms4732_hw1_data/cathedral.jpg', 'coms4732_hw1_data/monastery.jpg', 'coms4732_hw1_data/tobolsk.jpg']

# # for align_image_file in simple_align_image_files:
# for align_image_file in simple_align_image_files:
#     output_path = 'output/' + align_image_file.split('/')[-1].split('.')[0] + '_simple_aligned.jpg'
    
#     process_image(align_image_file, output_path, 'simple_alignment')

pyramid_align_image_files = [
    # 'coms4732_hw1_data/cathedral.jpg',
    # 'coms4732_hw1_data/monastery.jpg',
    # 'coms4732_hw1_data/tobolsk.jpg',
    # 'coms4732_hw1_data/church.tif',
    # 'coms4732_hw1_data/emir.tif',
    'coms4732_hw1_data/harvesters.tif',
    # 'coms4732_hw1_data/icon.tif',
    # 'coms4732_hw1_data/italil.tif',
    # 'coms4732_hw1_data/lastochikino.tif',
    # 'coms4732_hw1_data/lugano.tif',
    # 'coms4732_hw1_data/melons.tif',
    # 'coms4732_hw1_data/self_portrait.tif',
    # 'coms4732_hw1_data/siren.tif',
    # 'coms4732_hw1_data/three_generations.tif'
]

for align_image_file in pyramid_align_image_files:
    output_path = 'output/' + align_image_file.split('/')[-1].split('.')[0] + '_pyramid_aligned.jpg'
    
    process_image(align_image_file, output_path, 'pyramid_alignment')