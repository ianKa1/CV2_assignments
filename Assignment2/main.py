from PIL import Image, ImageDraw
from harris import get_harris_corners, dist_SSD
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from skimage.transform import resize
import os


IMAGE_NAME = 'tissue'

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

def visualize_matches(image1, image2, coords1, coords2, matches, output_path, gap=50):
    """
    Visualize feature matches between two images.

    Args:
        image1: PIL Image object (first image)
        image2: PIL Image object (second image)
        coords1: corner coordinates in image1, shape (2, n1) where coords[0] = ys, coords[1] = xs
        coords2: corner coordinates in image2, shape (2, n2) where coords[0] = ys, coords[1] = xs
        matches: array of shape (N, 2) where matches[:, 0] are indices into coords1,
                 matches[:, 1] are indices into coords2
        output_path: path to save the visualization
        gap: spacing between the two images (default: 50)
    """
    # Get dimensions
    w1, h1 = image1.size
    w2, h2 = image2.size

    # Create a new image that can hold both images with a gap
    combined_width = w1 + w2 + gap
    combined_height = max(h1, h2)
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste both images with gap
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (w1 + gap, 0))

    # Draw on the combined image
    draw = ImageDraw.Draw(combined_image)

    # Generate unique colors for each match using colormap
    num_matches = len(matches)
    cmap = plt.get_cmap('hsv')

    # Draw matches
    for i, match in enumerate(matches):
        idx1, idx2 = match

        # Get coordinates from coords arrays
        y1, x1 = coords1[0, idx1], coords1[1, idx1]
        y2, x2 = coords2[0, idx2], coords2[1, idx2]

        # Offset x2 by the width of the first image plus gap
        x2_offset = x2 + w1 + gap

        # Generate unique color for this match
        color_normalized = cmap(i / num_matches)[:3]  # Get RGB, ignore alpha
        color = tuple(int(c * 255) for c in color_normalized)

        # Draw circles at the matched points with unique color
        draw.ellipse([x1-3, y1-3, x1+3, y1+3], fill=color, outline=color)
        draw.ellipse([x2_offset-3, y2-3, x2_offset+3, y2+3], fill=color, outline=color)

        # Draw line connecting the matched points with the same unique color
        draw.line([(x1, y1), (x2_offset, y2)], fill=color, width=2)

    combined_image.save(output_path)
    return combined_image

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

def get_corner_features(image, coords, patch_size=40, out_size=8):
    H, W = image.shape
    half = patch_size // 2

    ys, xs = coords
    features = []
    valid_coords = []

    for y, x in zip(ys, xs):
        y = int(y)
        x = int(x)

        if y - half < 0 or y + half >= H or x - half < 0 or x + half >= W:
            continue

        patch = image[y-half:y+half+1, x-half:x+half+1]

        downsampled_patch = resize(patch, (out_size, out_size), anti_aliasing=True)

        d = downsampled_patch.flatten()

        d = d - d.mean()
        d = d / (d.std() + 1e-6)

        features.append(d)
        valid_coords.append((y, x))

    if len(features) == 0:
        return np.zeros((0, out_size * out_size), dtype=np.float32), np.zeros((2, 0), dtype=int)

    features = np.stack(features, axis=0)                # (N_valid, 64)
    valid_coords = np.array(valid_coords).T 
    return features, valid_coords

def compute_distance_matrix(features1, features2):
    diff = features1[:, None, :] - features2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist

def match_features_nndr(features1, features2, plot_hist=True):
    THRESHOLD = 0.5
    dist = compute_distance_matrix(features1, features2)
    N1 = dist.shape[0]

    sorted_idx = np.argsort(dist, axis=1)
    nn1_idx = sorted_idx[:, 0]
    nn2_idx = sorted_idx[:, 1]
    nn1_dist = dist[np.arange(N1), nn1_idx]
    nn2_dist = dist[np.arange(N1), nn2_idx]

    ratios = nn1_dist / (nn2_dist + 1e-6)
    mask = ratios < THRESHOLD
    matches = np.column_stack((np.arange(N1)[mask], nn1_idx[mask]))

    if plot_hist:
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=50, alpha=0.75)
        plt.axvline(THRESHOLD, color='r', linestyle='--', linewidth=2)
        plt.xlabel('NNDR (d1 / d2)')
        plt.ylabel('Count')
        plt.title(f'{IMAGE_NAME} NNDR Histogram (THRESHOLD = {THRESHOLD}) using SSD')
        plt.savefig(f'results/{IMAGE_NAME}/{IMAGE_NAME}_nndr_histogram.jpg')
        plt.close()

    # Return matches along with nn1_idx, nn2_idx, and ratios for visualization
    return matches, nn1_idx, nn2_idx, ratios

def visualize_top_matches(image1, image2, coords1, coords2, nn1_idx, nn2_idx, ratios, output_path, top_k=5, patch_size=40):
    """
    Visualize the top K best feature matches showing img1 feature, 1NN from img2, and 2NN from img2.

    Args:
        image1, image2: grayscale numpy arrays
        coords1, coords2: corner coordinates, shape (2, n)
        nn1_idx: indices of 1st nearest neighbors in img2 for each feature in img1
        nn2_idx: indices of 2nd nearest neighbors in img2 for each feature in img1
        ratios: NNDR ratios for each feature in img1
        output_path: path to save visualization
        top_k: number of best matches to visualize (default: 5)
        patch_size: size of patch to extract (default: 40)
    """
    # Find top K matches with smallest ratios
    best_indices = np.argsort(ratios)[:top_k]

    half = patch_size // 2
    H1, W1 = image1.shape
    H2, W2 = image2.shape

    # Create figure with top_k rows and 3 columns
    fig, axes = plt.subplots(top_k, 3, figsize=(9, 3 * top_k))
    if top_k == 1:
        axes = axes.reshape(1, -1)

    for row_idx, feat_idx in enumerate(best_indices):
        # Get coordinates for img1 feature
        y1, x1 = int(coords1[0, feat_idx]), int(coords1[1, feat_idx])

        # Get 1NN and 2NN from img2
        nn1_feat_idx = nn1_idx[feat_idx]
        nn2_feat_idx = nn2_idx[feat_idx]

        y2_nn1, x2_nn1 = int(coords2[0, nn1_feat_idx]), int(coords2[1, nn1_feat_idx])
        y2_nn2, x2_nn2 = int(coords2[0, nn2_feat_idx]), int(coords2[1, nn2_feat_idx])

        # Extract patches (with boundary checking)
        # Patch from img1
        y1_start, y1_end = max(0, y1-half), min(H1, y1+half)
        x1_start, x1_end = max(0, x1-half), min(W1, x1+half)
        patch1 = image1[y1_start:y1_end, x1_start:x1_end]

        # 1NN patch from img2
        y2_nn1_start, y2_nn1_end = max(0, y2_nn1-half), min(H2, y2_nn1+half)
        x2_nn1_start, x2_nn1_end = max(0, x2_nn1-half), min(W2, x2_nn1+half)
        patch2_nn1 = image2[y2_nn1_start:y2_nn1_end, x2_nn1_start:x2_nn1_end]

        # 2NN patch from img2
        y2_nn2_start, y2_nn2_end = max(0, y2_nn2-half), min(H2, y2_nn2+half)
        x2_nn2_start, x2_nn2_end = max(0, x2_nn2-half), min(W2, x2_nn2+half)
        patch2_nn2 = image2[y2_nn2_start:y2_nn2_end, x2_nn2_start:x2_nn2_end]

        # Display patches
        axes[row_idx, 0].imshow(patch1, cmap='gray')
        axes[row_idx, 0].set_title(f'Img1 Feature {row_idx+1}')
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(patch2_nn1, cmap='gray')
        axes[row_idx, 1].set_title(f'1NN (NNDR={ratios[feat_idx]:.3f})')
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(patch2_nn2, cmap='gray')
        axes[row_idx, 2].set_title(f'2NN')
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

os.makedirs(f"results/{IMAGE_NAME}", exist_ok=True)

image1 = Image.open(f'images/{IMAGE_NAME}1.jpg')
image2 = Image.open(f'images/{IMAGE_NAME}2.jpg')
image1_gray = convert_to_grayscale(image1)
image1_gray = np.array(image1_gray)
image2_gray = convert_to_grayscale(image2)  
image2_gray = np.array(image2_gray)

# Step 1: Get Harris corners
h1, coords1 = get_harris_corners(image1_gray)
h2, coords2 = get_harris_corners(image2_gray)
print(coords1.shape, coords2.shape)

# Visualize Harris corners on original images
visualize_corners(image1, coords1, f'results/{IMAGE_NAME}/{IMAGE_NAME}1_corners_STEP1.jpg', 1)
visualize_corners(image2, coords2, f'results/{IMAGE_NAME}/{IMAGE_NAME}2_corners_STEP1.jpg', 1)

# Step 2: Non-maximum suppression
coords_nms1 = nms_simple(h1)
coords_nms2 = nms_simple(h2)
print(coords_nms1.shape, coords_nms2.shape)

# Visualize NMS corners on original images
visualize_corners(image1, coords_nms1, f'results/{IMAGE_NAME}/{IMAGE_NAME}1_nms_STEP2.jpg', 3)
visualize_corners(image2, coords_nms2, f'results/{IMAGE_NAME}/{IMAGE_NAME}2_nms_STEP2.jpg', 3)

# Step 3: Get corner features
features1, valid_coords1 = get_corner_features(image1_gray, coords_nms1)
features2, valid_coords2 = get_corner_features(image2_gray, coords_nms2)
print(features1.shape, features2.shape)

# Step 4: Match features
matches, nn1_idx, nn2_idx, ratios = match_features_nndr(features1, features2)
print(matches.shape)

# Step 5: Visualize matches
visualize_matches(image1, image2, valid_coords1, valid_coords2, matches, f'results/{IMAGE_NAME}/{IMAGE_NAME}_matches_visualization.jpg')

# Step 6: Visualize top 5 best matches
visualize_top_matches(image1_gray, image2_gray, valid_coords1, valid_coords2, nn1_idx, nn2_idx, ratios,
                      f'results/{IMAGE_NAME}/{IMAGE_NAME}_top5_matches.jpg', top_k=5)


