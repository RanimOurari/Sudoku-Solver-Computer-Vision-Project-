import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import read_image, show_image


def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """
    
    edges = cv2.Canny(image, threshold1=175, threshold2=175)
    return edges
    


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlighted_edges (np.array): binary mask of shape [H, W]
    """
    kernel = np.ones((3, 3), np.uint8)
    highlighted_edges = cv2.dilate(edges, kernel, iterations=8)
    return highlighted_edges



def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (list of np.array): list of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours



def get_max_contour(contours):
    """
    Args:
        contours (list of np.array): list of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour



def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """

    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]
    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])
    
    return ordered_corners
    


def find_corners(contour, epsilon=0.02):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        epsilon (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """

    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)

    if len(corners) != 4:
        corners = np.vstack([corners, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])])[:4]

    ordered_corners = order_corners(corners)

    return ordered_corners



def rescale_image(image, scale=0.42):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """

    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    new_dimensions = (new_width, new_height)
    rescaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    rescaled_image = np.clip(rescaled_image, 0, 255).astype(np.uint8)

    return rescaled_image
    


def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """

    kernel_size = int(6 * sigma) | 1  
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    return blurred_image



def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    distance = np.linalg.norm(point1 - point2)
    return distance



def frontalize_image(image, ordered_corners):
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    top_left, top_right, bottom_right, bottom_left = ordered_corners
    width_top = distance(top_left, top_right)
    width_bottom = distance(bottom_left, bottom_right)
    height_left = distance(top_left, bottom_left)
    height_right = distance(top_right, bottom_right)

    side = int(max(np.mean([width_top, width_bottom]), np.mean([height_left, height_right])))

    destination_points = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype=np.float32)

    source_points = ordered_corners.astype(np.float32)
    transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_image = cv2.warpPerspective(image, transform_matrix, (side, side))
    assert warped_image.shape[0] == warped_image.shape[1], "Height and width of the warped image must be equal"
    return warped_image



def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, _ = pipeline(sudoku_image)

        show_image(frontalized_image, axis=axis, as_gray=True)
