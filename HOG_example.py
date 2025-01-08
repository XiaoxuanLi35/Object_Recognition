import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread
import os
from pathlib import Path
# import numpy as np #np.histogramdd(image.reshape(-1,3), bins=4)[0].flatten() color feature vector

def process_image_with_hog(image_path, output_dir):
    # Read the image
    image = imread(image_path)
    
    # Calculate HOG features
    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )
    breakpoint()
    # Create visualization - only for HOG image
    plt.figure(figsize=(1.28, 1.28), dpi=100)  # 1.28 inches * 100 dpi = 128 pixels
    
    # Rescale HOG image intensity for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    plt.axis('off')
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    
    # Save the result
    output_filename = f"hog_{Path(image_path).stem}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # Close figure to free memory

def batch_process_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_dir, filename)
            try:
                process_image_with_hog(image_path, output_dir)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Usage example #edit the path
if __name__ == "__main__":
    input_directory = 'C:/Users/myname/Desktop/Semester 3/CS6123/project_3/spriters/Forage'
    output_directory = os.path.join('C:/Users/myname/Desktop/Semester 3/CS6123/project_3/scikit-image/Forage', 'HOG')
    
    batch_process_images(input_directory, output_directory)
