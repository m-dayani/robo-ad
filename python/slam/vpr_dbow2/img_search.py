import cv2
import numpy as np
import glob
import imageproc  # The compiled C++ module


if __name__ == "__main__":
    images_path = "data/images"
    images = glob.glob(images_path + '/*.png')
    images = sorted(images)
    
    # Create an instance of the C++ class
    processor = imageproc.ImageProcessor()

    for i, image_path in enumerate(images):
        # Load an image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Pass the in-memory image to C++
        result = processor.process_image(image)

        print(f"Processed image {i}, value: {result}")
