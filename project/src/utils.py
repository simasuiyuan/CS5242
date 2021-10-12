import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Callable, Any, Dict
import cv2

def ismember(A, B):
    return [np.sum(a == B) for a in A ]


class image_utils():
    def __init__(self):
        self.is_gray: bool=True

    def plot_image(
        self,
        image: np.ndarray,
        figsize: tuple = (10, 10),
    ) -> None:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if self.is_gray:     
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        plt.show()
        plt.close()

    def get_image(
        self,
        file_path: Union[str, pathlib.Path],
        image_scale: int = cv2.COLOR_BGR2RGB,
        is_gray=False,
        cmap_type:str=None
    ) -> np.ndarray:
        pathlib.Path(file_path)
        self.is_gray = is_gray
        image = cv2.imread(file_path.as_posix())
        image = cv2.cvtColor(image, image_scale)
        # if self.is_gray:
        #     image = cv2.imread(file_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        # else:
        #     image = cv2.imread(file_path.as_posix())
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image