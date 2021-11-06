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
        title: str = "",
        figsize: tuple = (10, 10),
    ) -> None:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if self.is_gray:     
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        plt.title(title, fontsize=10)
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

    def show_prob_pimples(self, p):
        p=p.data.squeeze().numpy()
        ft=15
        label = ('Not Skin', 'Normal', 'Pustule', 'Whitehead', 'Blackhead', 'Cyst',)
        #p=p.data.squeeze().numpy()
        y_pos = np.arange(len(p))*1.2
        target=2
        width=0.9
        col= 'blue'
        #col='darkgreen'

        plt.rcdefaults()
        fig, ax = plt.subplots()

        # the plot
        ax.barh(y_pos, p, width , align='center', color=col)

        ax.set_xlim([0, 1.3])
        #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

        # y label
        ax.set_yticks(y_pos)
        ax.set_yticklabels(label, fontsize=ft)
        ax.invert_yaxis()  
        #ax.set_xlabel('Performance')
        #ax.set_title('How fast do you want to go today?')

        # x label
        ax.set_xticklabels([])
        ax.set_xticks([])
        #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
        #ax.set_xticks(x_pos)
        #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(4)


        for i in range(len(p)):
            str_nb="{0:.4f}".format(p[i])
            ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transData, color= col,fontsize=ft)



        plt.show()