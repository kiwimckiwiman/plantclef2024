import random
import os
import cv2
import numpy as np
from itertools import zip_longest

class patch():
    def __init__(self, top_left, top_right, bottom_right, bottom_left):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left

    def width(self):
        return self.top_right[0] - self.top_left[0]
    
    def height(self):
        return self.bottom_right[1] - self.top_right[1]
    
    def coords(self):
          print(f'top left: {self.top_left}, top right: {self.top_right}, bottom right: {self.bottom_right}, bottom left: {self.bottom_left}')

class patches():
    def __init__(self, height, width, patch_no, min_size):
        self.height = height
        self.width = width
        self.patch_no = patch_no
        self.min_size = min_size
        self.all_patches = []
        self.create_patches()
        
    def get_orientation(self):
        bool_list = [True]
        for i in range(1, self.patch_no - 1):
            bool_list.append(bool_list[-1] if i % 2 == 0 else not bool_list[-1])
        k = random.randint(0,100)%2
        if k == 0:
        	bool_list = list(map(lambda x: not x, bool_list))
        return bool_list
    
    def create_patches(self):
        curr = []
        curr.append(patch([0,0], [self.width, 0], [self.width, self.height], [0, self.height]))
        orientations = self.get_orientation()
        for i in range(0, self.patch_no - 1):
            orientation = orientations[i]
            curr_patch = curr[i]
            top_left = curr_patch.top_left
            top_right = curr_patch.top_right
            bottom_right = curr_patch.bottom_right
            bottom_left = curr_patch.bottom_left
            if orientation == True:
                if(curr_patch.width() >= self.min_size*2):
                    x = random.randrange(round((curr_patch.width() - self.min_size)/2), round((curr_patch.width() + self.min_size)/2), round(self.min_size/3))
                    x = round(x/32)*32
                    x_split_1 = [x, curr_patch.top_left[1]]
                    x_split_2 = [x, curr_patch.bottom_left[1]]                      
                                        
                    patch_1 = patch(top_left, x_split_1, x_split_2, bottom_left)
                    patch_2 = patch(x_split_1, top_right, bottom_right, x_split_2)
                    
                    curr.append(patch_1)
                    curr.append(patch_2)
            else:
                if(curr_patch.height() >= self.min_size*2):
                    y = random.randrange(round((curr_patch.height() - self.min_size)/2), round((curr_patch.height() + self.min_size)/2), round(self.min_size/3))                  
                    y = round(y/32)*32
                    
                    y_split_1 = [curr_patch.top_left[0], y]
                    y_split_2 = [curr_patch.top_right[0], y]                
                                        
                    patch_1 = patch(top_left, top_right, y_split_2, y_split_1)
                    patch_2 = patch(y_split_1, y_split_2, bottom_right, bottom_left)
                    
                    curr.append(patch_1)
                    curr.append(patch_2)
        del curr [0:(self.patch_no-1)]
        self.all_patches = curr

def fit_to_patch(image, patch):
    patch_width = patch.width()
    patch_height = patch.height()
    image_height, image_width = image.shape[:2]
    
    if image_height >= patch_height:
        if image_width >= patch_width:
            resized_image = image
        elif image_width < patch_width:
            new_height = round((image_height * patch_width)/image_width)
            resized_image = cv2.resize(image, (patch_width, new_height))
    elif image_height < patch_height:
        new_width = round((image_width * patch_height)/image_height)
        resized_image = cv2.resize(image, (new_width, patch_height))
            
    resized_image_height, resized_image_width = resized_image.shape[:2]
    x1 = round(abs(resized_image_width - patch_width)/2)
    x2 = x1 + patch_width
    y1 = round(abs(resized_image_height - patch_height)/2)
    y2 = y1 + patch_height
    cropped_image = resized_image[y1:y2, x1:x2]                                  
    
    return cropped_image

def generate_laplacian(img, pyr_lap, level):
    g_pyr = [img]
    g = img
    for i in range(0, level):
        g = cv2.pyrDown(g)
        g_pyr.append(g)
    
    l_pyr = [g_pyr[level-1]]
    
    for i in range(level-1, 0, -1):
        l = cv2.subtract(g_pyr[i-1], cv2.pyrUp(g_pyr[i]))
        l_pyr.append(l)
    return l_pyr    

def generate_patched_img(train_path, height, width, min_class, max_class, min_size):
    pyramid_level = 4
    training_imgs = train_path
    imgs_count = random.randint(min_class,max_class)

    classes = os.listdir(training_imgs)
    
    selected = random.sample(classes, imgs_count)
    imgs = []
    for i in selected:
        class_path = os.path.join(training_imgs, i)
        img_path = os.path.join(class_path, random.choice(os.listdir(class_path)))
        img = cv2.imread(img_path)
        imgs.append(img)
        
    patch_dims = patches(height, width, imgs_count, min_size)
    
    all_laplacians = []
    for i in range(0, imgs_count):
        patch = fit_to_patch(imgs[i], patch_dims.all_patches[i])
        pyramid = []
        all_laplacians.append(generate_laplacian(patch, pyramid, pyramid_level))
    
    transposed_laplacian = []
    for i in range(0, len(all_laplacians[0])):
        laplacian_wise_patch = []
        for j in range(0, len(all_laplacians)):
            laplacian_wise_patch.append(all_laplacians[j][i])
        transposed_laplacian.append(laplacian_wise_patch)

    i = pyramid_level
    laplacian_canvases = []
    for lap_level in transposed_laplacian:
        canvas_height = int(height/pow(2, i-1))
        canvas_width =  int(width/pow(2, i-1))
        canvas = np.full((canvas_height , canvas_width, 3), (255, 255, 255), dtype=np.uint8)
        for k in range(0, len(patch_dims.all_patches)):
            x = round(patch_dims.all_patches[k].top_left[0]/pow(2, i-1))
            y =  round(patch_dims.all_patches[k].top_left[1]/pow(2, i-1))
            
            patch = lap_level[k]
            patch_height, patch_width, _ = patch.shape
            
            canvas[y:y + patch_height, x:x + patch_width] = patch
        laplacian_canvases.append(canvas)
        i -= 1   
        
    
    blended_img = laplacian_canvases[0]
    for i in range(1, len(laplacian_canvases)):
        blended_img = cv2.pyrUp(blended_img)
        blended_img = cv2.add(blended_img, laplacian_canvases[i])
    cv2.imshow("pls", blended_img)
    cv2.waitKey(0)
    return canvas, selected        


# IMPORTANT: patch generation is buggy past 6 images, ideal range is 1-4
# experiment with min size till it gives an image
# (e.g., if desired patch size is 500x500, min_size should be kept at 100)
# pls make sure width and height are multiples of 32, needs to be divisible
# by 32 for 4 levels of laplacian.
# for n levels of laplacian, widht/height and rounded rand x and y values = 2^(n+1)

# Parameters:
TRAIN_PATH = r"C:\Users\ASUS\Desktop\plantclef\image_samples\samples"
PATCH_HEIGHT = 512
PATCH_WIDTH = 512
MIN_CLASS = 3
MAX_CLASS = 4
MIN_SIZE = 150

args = TRAIN_PATH, PATCH_HEIGHT, PATCH_WIDTH, MIN_CLASS, MAX_CLASS, MIN_SIZE
img, classes = generate_patched_img(*args)
cv2.imshow("_".join(classes), img)
cv2.waitKey(0)