import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load images from a directory
image_folder = 'enhanced'
images = [cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_GRAYSCALE) for file in os.listdir(image_folder)]

# Flatten images into 1D arrays
flattened_images = np.array([img.flatten() for img in images])

# Perform PCA
pca = PCA()
pca.fit(flattened_images)

# Project all the images along the eigenvectors
projected = pca.transform(flattened_images)

# Find the maximum and minimum values along each eigenvector
max_vals = projected.max(axis=0)
min_vals = projected.min(axis=0)

# Generate new images
n_new_images = 10
generated_images = []
for _ in range(n_new_images):
    random_coefficients = np.random.uniform(min_vals, max_vals)
    new_image = np.dot(random_coefficients, pca.components_) + pca.mean_
    generated_images.append(new_image)

# Reshape the images and convert back to 2D
generated_images = np.array(generated_images).reshape((-1, *images[0].shape))

# Save the generated images
output_folder = 'pca_w_clahe'
os.makedirs(output_folder, exist_ok=True)

for i, img in enumerate(generated_images):
    cv2.imwrite(os.path.join(output_folder, f'pca_image_{i}.jpg'), img)

# Display the generated images
#plt.figure(figsize=(8, 8))
#for i in range(n_new_images):
#    plt.subplot(1, n_new_images, i + 1)
#    plt.imshow(generated_images[i], cmap='gray')
#    plt.axis('off')
#plt.show()
#
