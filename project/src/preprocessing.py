import numpy as np



def imagecrop(
    image:np.ndarray,
    top:int = 110,
    right:int = 120,
    down:int = 120,
    left:int = 120,
) -> np.ndarray:
    w,h = image.shape[:2]
    return image[top:((w-down)+top), right:((h-left)+right)]

def imadjust(x,min,max,new_min,new_max,gamma=1):
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.
    y = (((x - min) / (max - min)) ** gamma) * (new_max - new_min) + new_min
    return y