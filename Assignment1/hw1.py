# COMS4732: Project 1 starter Python code
# Taken from: CS180 at UC Berkeley

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale

def compute_ssd(shifted, reference):
    """Compute SSD only on overlapping region to avoid edge artifacts."""
    # Find overlapping region (where both images have valid data)
    # For simplicity, we'll compute on a cropped central region
    h, w = reference.shape
    crop = min(50, h // 10, w // 10)  # Crop border to avoid wrap-around effects
    if crop > 0:
        ref_crop = reference[crop:-crop, crop:-crop]
        shift_crop = shifted[crop:-crop, crop:-crop]
        return np.sum((shift_crop - ref_crop) ** 2)
    else:
        return np.sum((shifted - reference) ** 2)

def align(channel, reference, search_range=30):
    """
    Align a channel to a reference channel using L2 norm (SSD).
    Uses multiscale approach for efficiency and accuracy.
    
    Args:
        channel: The channel to align (2D array)
        reference: The reference channel (2D array)
        search_range: Maximum pixel displacement to search
    
    Returns:
        Aligned channel
    """
    # Multiscale alignment: start with downscaled version for speed
    if min(channel.shape[0], channel.shape[1]) > 300:
        # Downscale for coarse alignment
        scale = 0.25
        ref_small = rescale(reference, scale, anti_aliasing=True, preserve_range=True)
        ch_small = rescale(channel, scale, anti_aliasing=True, preserve_range=True)
        
        # Coarse search on downscaled image
        best_shift_coarse = (0, 0)
        best_score = float('inf')
        coarse_range = min(15, search_range // 2)
        
        for dy in range(-coarse_range, coarse_range + 1):
            for dx in range(-coarse_range, coarse_range + 1):
                shifted = np.roll(np.roll(ch_small, dy, axis=0), dx, axis=1)
                score = compute_ssd(shifted, ref_small)
                if score < best_score:
                    best_score = score
                    best_shift_coarse = (int(np.round(dy / scale)), int(np.round(dx / scale)))
        
        # Refine search around the coarse result
        refine_range = 8
        start_dy, start_dx = best_shift_coarse
    else:
        start_dy, start_dx = 0, 0
        refine_range = search_range
    
    # Fine search on full resolution around coarse result
    best_shift = (start_dy, start_dx)
    best_score = float('inf')
    
    for dy in range(start_dy - refine_range, start_dy + refine_range + 1):
        for dx in range(start_dx - refine_range, start_dx + refine_range + 1):
            shifted = np.roll(np.roll(channel, dy, axis=0), dx, axis=1)
            score = compute_ssd(shifted, reference)
            if score < best_score:
                best_score = score
                best_shift = (dy, dx)
    
    print(f"Best shift found: dy={best_shift[0]}, dx={best_shift[1]}, score={best_score:.2e}")
    
    # Apply the best shift
    aligned = np.roll(np.roll(channel, best_shift[0], axis=0), best_shift[1], axis=1)
    return aligned

def process_image(imname, output_name, border_size=30, search_range=30):
    """
    Process a single image: align channels and save the result.
    
    Args:
        imname: Input image filename
        output_name: Output image filename
        border_size: Size of border to remove after alignment
        search_range: Maximum pixel displacement to search for alignment
    """
    print(f"\n{'='*60}")
    print(f"Processing: {imname}")
    print(f"{'='*60}")
    
    # read in the image
    im = skio.imread(imname)
    
    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    
    # separate color channels (BGR format)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    # Align the images using L2 norm (align before removing borders)
    print("Aligning green channel...")
    ag_aligned = align(g, b, search_range=search_range)
    print("Aligning red channel...")
    ar_aligned = align(r, b, search_range=search_range)
    b_aligned = b
    
    # Remove borders after alignment
    # Use larger border to account for alignment shifts
    b_final = b_aligned[border_size:-border_size, border_size:-border_size]
    ag_final = ag_aligned[border_size:-border_size, border_size:-border_size]
    ar_final = ar_aligned[border_size:-border_size, border_size:-border_size]
    
    # create a color image (RGB format)
    im_out = np.dstack([ar_final, ag_final, b_final])
    
    # Clip values to valid range [0, 1]
    im_out = np.clip(im_out, 0, 1)
    
    # Convert to uint8 for saving (0-255 range)
    im_out_uint8 = (im_out * 255).astype(np.uint8)
    
    # save the image
    skio.imsave(output_name, im_out_uint8)
    print(f"Image saved successfully as {output_name}")

# Process multiple images
image_files = [
    ('coms4732_hw1_data/cathedral.jpg', 'cathedral_aligned.jpg'),
    ('coms4732_hw1_data/monastery.jpg', 'monastery_aligned.jpg'),
    ('coms4732_hw1_data/tobolsk.jpg', 'tobolsk_aligned.jpg')
]

for imname, output_name in image_files:
    process_image(imname, output_name)

print(f"\n{'='*60}")
print("All images processed successfully!")
print(f"{'='*60}")