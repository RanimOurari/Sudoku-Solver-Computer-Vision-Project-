from frontalization import find_contours, find_corners, find_edges, frontalize_image, gaussian_blur, get_max_contour, highlight_edges
from pipeline import Pipeline
import numpy as np
from recognition import get_sudoku_cells


"""
Define a dictionary of cell coordinates for different images.
Each key is an image filename, and each value is a dictionary of digit positions.
The format is as follows:

CELL_COORDINATES = {
    "image_0.jpg": {1: (0, 0), 2: (1, 1)},
    "image_2.jpg": {1: (2, 3), 3: [(2, 1), (0, 4)], 9: (5, 6)}
}
"""

CELL_COORDINATES = {
    "image_2.jpg": {
        1: (0, 7),
        2: (7, 7),
        4: (5, 4),
        5: (7, 1),
        6: (2, 5),
        7: (5, 2),
        8: (2, 0),
        
    },
    "image_6.jpg": {
        1: (0, 0),
        2: (4, 2),
        4: (4, 6),
        6: (0, 8),
        8: (2, 7),
        9: (2, 6),
    },


    "image_7.jpg": {
        7: (2, 4),
    },


    "image_4.jpg": {
        9: (1, 4),
        3: (1,7),
    },


    "image_9.jpg": {
        3: (2,0 ),
        5: (4,1),
    },
}


def get_template_pipeline():
    pipeline = Pipeline(functions=[gaussian_blur, 
                                   find_edges, 
                                   highlight_edges, 
                                   find_contours, 
                                   get_max_contour, 
                                   find_corners, 
                                   frontalize_image,  
                                   gaussian_blur,                                                                   
                                   get_sudoku_cells,
                                   ],
                    parameters={"gaussian_blur": {"sigma": 0.5}, 
                                "find_corners": {"epsilon": 0.01}}) 

    return pipeline
