import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import random
from  sklearn.mixture  import GaussianMixture as GMM

def blend_images(image_A, image_B, coordinates_B_on_A):
    # Ensure that both images are of the same type and have the same number of channels
    # if image_A.shape[2] == 1:
    image_A = cv2.cvtColor(image_A, cv2.COLOR_GRAY2BGR)
    # if image_B.shape[2] == 1:
    image_B = cv2.cvtColor(image_B, cv2.COLOR_GRAY2BGR)

    # Create a mask of the region to be replaced
    mask = 255 * np.ones(image_B.shape, image_B.dtype)

    # Perform seamless cloning
    blended_image = cv2.seamlessClone(image_B, image_A, mask, coordinates_B_on_A, cv2.NORMAL_CLONE)

    return blended_image

import cv2
import numpy as np

def create_binary_mask(image_A):
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY)
    unsigned_img=(image_A*255).astype(np.uint8)
    # Compute Otsu's threshold
    _, binary_mask = cv2.threshold(unsigned_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_mask
def generate_gaussian_image(size, sigma):
    # Generate a grid of indices
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # Create a 2D Gaussian function
    d = np.sqrt(x*x + y*y)
    sigma, mu = sigma, 0.0
    gaussian = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))

    # Rescale to 0-255 and convert to uint8
    gaussian = (255 * gaussian).astype(np.uint8)

    return gaussian


def remove_background(image_A):
    # Create a binary mask using Otsu's method
    binary_mask = create_binary_mask(image_A)

    # Find the minimum pixel intensity in the original image
    min_intensity = 0

    # Apply the binary mask to make the background black
    final_image = np.where(binary_mask[:, :] == 0, 0, image_A)

    return (final_image).astype(np.uint8)



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
n_new_images = 100
generated_images = []
for _ in range(n_new_images):
    random_coefficients = np.random.uniform(min_vals, max_vals)
    new_image = np.dot(random_coefficients, pca.components_) + pca.mean_
    generated_images.append(new_image)

# Reshape the images and convert back to 2D
generated_images = np.array(generated_images).reshape((-1, *images[0].shape))

df = pd.read_csv('jsrt_metadata.csv')
nodule_images=df[df["state"]!="non-nodule"]
x_s=nodule_images["x"]
y_s=nodule_images["y"]
size=nodule_images["size"]
gmm = GMM(n_components=4, random_state=0).fit(np.array([x_s,y_s,size]).T)
region_proposals=gmm.sample(n_new_images)[0]
non_nodule_images=df[df["state"]=="non-nodule"]["study_id"]
new_images=random.choices(non_nodule_images.values.tolist(), k=n_new_images)


# Save the generated images
output_folder = 'pca_w_clahe'
os.makedirs(output_folder, exist_ok=True)
pca_output_path = os.path.join( output_folder, 'pca_output')
pca_masks_path = os.path.join( output_folder, 'masks')
os.makedirs( output_folder, exist_ok= True)
os.makedirs( pca_output_path, exist_ok= True)
os.makedirs( pca_masks_path, exist_ok= True)


for i, img in enumerate(generated_images):
    try:
        print((int(region_proposals[i,0]),int(region_proposals[i,1])))
        reshaped_nodule=cv2.resize(img,(8*int(region_proposals[i,2]),8*int(region_proposals[i,2])))
        nnimage=cv2.imread(os.path.join("images", new_images[i]), cv2.IMREAD_GRAYSCALE)
        px,py,s=int(region_proposals[i,0]),int(region_proposals[i,1]),int(region_proposals[i,2])
        s=s*4
        # nnimage = blend_images(nnimage,reshaped_nodule,(int(region_proposals[i,0]),int(region_proposals[i,1])))
        mask = cv2.GaussianBlur(remove_background(reshaped_nodule),(5, 5), 5)
        var=0.5
        mask=(0.5*np.max(nnimage[py-s:py+s,px-s:px+s])/np.max(mask))*mask
        nnimage[py-s:py+s,px-s:px+s] += np.multiply(generate_gaussian_image(2*s,var)/255,mask).astype(np.uint8)
        mask=np.multiply(generate_gaussian_image(2*s,var)/255,mask).astype(np.uint8)
        cv2.rectangle(nnimage,(px-2*s,py-2*s),(px+2*s,py+2*s),(0,255,0),3)
        cv2.imwrite( os.path.join( pca_masks_path, f'mask_{i}.jpg'), mask)

        cv2.imwrite(os.path.join(pca_output_path, f'generated_image_{i}.jpg'), nnimage)
    except Exception as e:
        print(e)

# Display the generated images
#plt.figure(figsize=(8, 8))
#for i in range(n_new_images):
#    plt.subplot(1, n_new_images, i + 1)
#    plt.imshow(generated_images[i], cmap='gray')
#    plt.axis('off')
#plt.show()
#
