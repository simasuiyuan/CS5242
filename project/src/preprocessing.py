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


def imagePaddingByShape(
    image:np.ndarray,
    shape:tuple,
    padding_mode = "constant" #"constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty"
) -> np.ndarray:
    w,h = image.shape[:2]
    if len(shape) == 2:
        pad = [(0, shape[0]-w), (0, shape[1]-h)]
    elif len(shape) == 3:
        pad = [(0, shape[0]-w), (0, shape[1]-h),(0,0)]
    else:
        pad = []
    return np.pad(image.copy(), pad, padding_mode)