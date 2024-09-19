# IMPORTANT: patch generation is buggy past 6 images, ideal range is 1-4
# experiment with min size till it gives an image
# (e.g., if desired patch size is 500x500, min_size should be kept at 100)
# the larger the image size, the better patch sizes 5-6 work
# labels are in array format e.g., [class1, class2, class3]
# if using pillow to view/use image: Image.fromarray(np.uint8(patch*255))

import patch_images_unblended as img_gen
import cv2
import os
import tqdm as tqdm
import time

# Parameters:
IMGS = r"C:\Users\ASUS\Desktop\plantclef\image_samples\samples" #path to the images you want to build patched img from
PATCH_HEIGHT = 1512
PATCH_WIDTH = 1512
MIN_CLASS = 4 # min number of classes present in image
MAX_CLASS = 8 # max number of classes present in image
MIN_SIZE = 150 # min width/height of image

FULL_SIZED = r"C:\Users\ASUS\Desktop\plantclef\image_samples\full_sized"
args = IMGS, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img_generator = img_gen.patched_img(*args)

for i in tqdm.tqdm(range(0,8)):
    time.sleep(0.1)
    patch, label = img_generator.generate_patched_img() # use this as many times to generate as many patches and labels
    img_name = "_".join(label) + ".jpg"
    cv2.imwrite(os.path.join(FULL_SIZED, img_name), patch)