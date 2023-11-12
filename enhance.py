import cv2
import sys
import numpy as np
import os

os.makedirs('enhanced', exist_ok = True)
img_folder = sys.argv[1]
imgs = os.listdir( img_folder)

for img_name in imgs:
    try:
        img = cv2.imread(f'{img_folder}/{img_name}')
        # converting to LAB color space
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(f'./enhanced/{img_name}', enhanced_img)
        # Stacking the original image with the enhanced image
        #result = np.hstack((img, enhanced_img))
        #cv2.imshow('Result', result)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    except Exception as e:
        print(e)

