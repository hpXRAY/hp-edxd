import cv2
import numpy as np
import time

# Create a black image with a white diagonal line (monochrome)
image_size = (192, 4000)
image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
cv2.line(image, (50, 50), (250, 250), 255, 2)

# Define the custom scaling function
def custom_scaling_function(x):
    scaling_factor = 1.0 + x * 0.005
    return scaling_factor

# Create an empty result image
result = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

# Apply the custom scaling transformation
for x in range(image_size[1]):
    scaling_factor = custom_scaling_function(x)
    
    src_x = x / scaling_factor
    
    if 0 <= src_x < image_size[1] - 1:
        x1 = int(src_x)
        x2 = x1 + 1
        alpha = src_x - x1
        result[:, x] = (1 - alpha) * image[:, x1] + alpha * image[:, x2]

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Display the transformed image
cv2.imshow('Transformed Image', result)
cv2.waitKey(0)

