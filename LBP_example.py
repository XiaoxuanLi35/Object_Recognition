import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import feature
import warnings

# Suppress the specific PIL warning
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')

# Define input and output paths
input_path = 'C:/Users/myname/Desktop/Semester 3/CS6123/project_3/spriters/Fishes'
output_base = 'C:/Users/myname/Desktop/Semester 3/CS6123/project_3/scikit-image/Fishes'

# Create LBP output folder
output_path = os.path.join(output_base, 'LBP')
if not os.path.exists(output_path):
    os.makedirs(output_path)

METHOD = 'uniform'
plt.rcParams['font.size'] = 9

def plot_circle(ax, center, radius, color):
    circle = plt.Circle(center, radius, facecolor=color, edgecolor='0.5')
    ax.add_patch(circle)

def plot_lbp_model(ax, binary_values):
    """Draw the schematic for a local binary pattern."""
    theta = np.deg2rad(45)
    R = 1
    r = 0.15
    w = 1.5
    gray = '0.5'
    
    plot_circle(ax, (0, 0), radius=r, color=gray)
    
    for i, facecolor in enumerate(binary_values):
        x = R * np.cos(i * theta)
        y = R * np.sin(i * theta)
        plot_circle(ax, (x, y), radius=r, color=str(facecolor))
    
    for x in np.linspace(-w, w, 4):
        ax.axvline(x, color=gray)
        ax.axhline(x, color=gray)
    
    ax.axis('image')
    ax.axis('off')
    size = w + 0.2
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

def process_image(image_path, output_path):
    # Read image
    with Image.open(image_path) as img:
        # Convert to RGBA first if image has transparency
        if img.mode == 'P':
            img = img.convert('RGBA')
        # Then convert to grayscale
        gray = img.convert('L')
        # Convert to numpy array
        gray_array = np.array(gray)
    
    # Calculate LBP
    radius = 1
    n_points = 8
    lbp = feature.local_binary_pattern(gray_array, n_points, radius, METHOD)
    
    # Create figure
    plt.figure(figsize=(6, 6))
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')
    
    # Save processed image
    output_filename = os.path.join(output_path, f'lbp_{os.path.basename(image_path)}')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process all images in the folder
for filename in os.listdir(input_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_path, filename)
        print(f"Processing image: {filename}")
        try:
            process_image(image_path, output_path)
            print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All images processing completed!")
