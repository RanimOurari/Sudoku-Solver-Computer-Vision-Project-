import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from frontalization import gaussian_blur
from utils import read_image, show_image
from const import NUM_CELLS, CELL_SIZE, SUDOKU_SIZE
from utils import load_templates

def resize_image(image, size):
    """
    Args:
        image (np.array): input image of shape [H, W]
        size (tuple): desired image size (width, height)
    Returns:
        resized_image (np.array): 8-bit (with range [0, 255]) resized image
    """
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    
    return resized_image


def binarize(image, **binarization_kwargs):
    """
    Args:
        image (np.array): input image
        binarization_kwargs (dict): dict of parameter values
    Returns:
        binarized_image (np.array): binarized image
    """
    
    maxval=255
    thresh=108
    _, binarized_image = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)

    return binarized_image


def crop_image(image, crop_factor):
    """Crop the central part of the image by a given crop factor."""
    size = image.shape[:2]
    cropped_size = (int(size[0] * crop_factor), int(size[1] * crop_factor))
    shift = ((size[0] - cropped_size[0]) // 2, (size[1] - cropped_size[1]) // 2)

    cropped_image = image[shift[0]:shift[0] + cropped_size[0], shift[1]:shift[1] + cropped_size[1]]
    return cropped_image

def center_digit(cell, background_color=255):
    """
    Center the digit in the Sudoku cell.
    """
    coords = np.column_stack(np.where(cell < 128)) 
    if len(coords) == 0:
        return cell  
    mean_y, mean_x = np.mean(coords, axis=0).astype(int)
    shift_y = cell.shape[0] // 2 - mean_y
    shift_x = cell.shape[1] // 2 - mean_x
    centered_cell = np.full_like(cell, background_color)
    y_start = max(0, shift_y)
    y_end = min(cell.shape[0], shift_y + cell.shape[0])
    x_start = max(0, shift_x)
    x_end = min(cell.shape[1], shift_x + cell.shape[1])
    
    centered_cell[y_start:y_end, x_start:x_end] = cell[max(0, -shift_y):min(cell.shape[0], cell.shape[0] - shift_y),max(0, -shift_x):min(cell.shape[1], cell.shape[1] - shift_x)]
    
    return centered_cell

def get_sudoku_cells(frontalized_image, crop_factor=0.64, binarization_kwargs={}):
    """
    Args:
        frontalized_image (np.array): frontalized sudoku image
        crop_factor (float): how much cell area we should preserve
        binarization_kwargs (dict): dict of parameter values for the binarization function
    Returns:
        sudoku_cells (np.array): array of num_cells x num_cells sudoku cells of shape [N, N, S, S]
    """
    resized_image = resize_image(frontalized_image, SUDOKU_SIZE)
    binarized_image = binarize(gaussian_blur(resized_image,0.42), **binarization_kwargs)
    sudoku_cells = np.zeros((NUM_CELLS, NUM_CELLS, *CELL_SIZE), dtype=np.uint8)
    cell_width = SUDOKU_SIZE[0] // NUM_CELLS
    
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            sudoku_cell = binarized_image[i*cell_width:(i+1)*cell_width, j*cell_width:(j+1)*cell_width]
            sudoku_cell = crop_image(sudoku_cell, crop_factor=crop_factor)
            sudoku_cell= center_digit(sudoku_cell)
            sudoku_cells[i, j] = resize_image(sudoku_cell, CELL_SIZE)
    
    return sudoku_cells

def is_empty(sudoku_cell, **kwargs):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        kwargs (dict): dict of parameter values for this function
    Returns:
        is_empty (bool): True or False depends on whether the Sudoku cell is empty or not
    """
    is_empty = False
    num_black_pixels = np.sum(sudoku_cell == 0)
    total_pixels = sudoku_cell.size
    percentage_black = (num_black_pixels / total_pixels) * 100
    if percentage_black < kwargs['threshold']:
        is_empty = True
    
    return is_empty


def get_digit_correlations(sudoku_cell, templates_dict, threshold=2):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
    Returns:
        correlations (np.array): an array of correlation coefficients between Sudoku cell and digit templates
    """
    correlations = np.zeros(9)
    
    if is_empty(sudoku_cell, threshold=threshold):
        return correlations
    
    sudoku_cell = cv2.normalize(sudoku_cell, None, 0, 255, cv2.NORM_MINMAX)  
    for digit, templates in templates_dict.items():
        resault1 = cv2.minMaxLoc(cv2.matchTemplate(sudoku_cell, templates[0], cv2.TM_CCOEFF_NORMED))
        resault2 = cv2.minMaxLoc(cv2.matchTemplate(sudoku_cell, templates[1], cv2.TM_CCOEFF_NORMED))
        if resault1[1] > resault2[1]:
            correlations[digit - 1] = resault1[1]
        else:
            correlations[digit - 1] = resault2[1]


    return correlations



def show_correlations(sudoku_cell, correlations):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    show_image(sudoku_cell, axis=axes[0], as_gray=True)
    
    colors = ['blue' if value < np.max(correlations) else 'red' for value in correlations]
    axes[1].bar(np.arange(1, 10), correlations, tick_label=np.arange(1, 10), color=colors)
    axes[1].set_title("Correlations")
    axes[1].set_xlabel("Digits")
    axes[1].set_ylabel("Correlation Coefficient")


def recognize_digits(sudoku_cells, templates_dict, threshold=2):
    """
    Args:
        sudoku_cells (np.array): np.array of the Sudoku cells of shape [N, N, S, S]
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
        threshold (float): empty cell detection threshold
    Returns:
        sudoku_matrix (np.array): a matrix of shape [N, N] with recognized digits of the Sudoku grid
    """
    sudoku_matrix = np.zeros(sudoku_cells.shape[:2], dtype=np.uint8)
    for i in range(sudoku_cells.shape[0]):
        for j in range(sudoku_cells.shape[1]):
            correlations = get_digit_correlations(sudoku_cells[i][j], templates_dict, threshold)
            if np.all(correlations == 0):
                sudoku_matrix[i, j] = 0
            else:
                sudoku_matrix[i, j] = np.argmax(correlations) + 1
    
    return sudoku_matrix


def show_recognized_digits(image_paths, pipeline, figsize=(16, 12), digit_fontsize=10):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for index, image_path in enumerate(tqdm(image_paths)):
        axis = axes[index // ncols][index % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, sudoku_cells = pipeline(sudoku_image)

        templates_dict = load_templates()
        sudoku_matrix = recognize_digits(sudoku_cells, templates_dict)

        show_image(frontalized_image, axis=axis, as_gray=True)
        
        frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
        for i in range(NUM_CELLS):
            for j in range(NUM_CELLS):
                axis.text((j + 1)*frontalized_cell_size[0] - int(0.3*frontalized_cell_size[0]),
                          i*frontalized_cell_size[1] + int(0.3*frontalized_cell_size[1]),
                          str(sudoku_matrix[i, j]), fontsize=digit_fontsize, c='r')


def show_solved_sudoku(frontalized_image, sudoku_matrix, sudoku_matrix_solved, digit_fontsize=20):
    show_image(frontalized_image, as_gray=True)

    frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            if sudoku_matrix[i, j] == 0:
                plt.text(j*frontalized_cell_size[0] + int(0.3*frontalized_cell_size[0]),
                         (i + 1)*frontalized_cell_size[1] - int(0.3*frontalized_cell_size[1]),
                         str(sudoku_matrix_solved[i, j]), fontsize=digit_fontsize, c='g')
