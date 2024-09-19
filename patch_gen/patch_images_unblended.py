import random
import os
import cv2
import numpy as np

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
                    
                    x_split_1 = [x, curr_patch.top_left[1]]
                    x_split_2 = [x, curr_patch.bottom_left[1]]                      
                                        
                    patch_1 = patch(top_left, x_split_1, x_split_2, bottom_left)
                    patch_2 = patch(x_split_1, top_right, bottom_right, x_split_2)
                    
                    curr.append(patch_1)
                    curr.append(patch_2)
            else:
                if(curr_patch.height() >= self.min_size*2):
                    y = random.randrange(round((curr_patch.height() - self.min_size)/2), round((curr_patch.height() + self.min_size)/2), round(self.min_size/3))                  
                    
                    y_split_1 = [curr_patch.top_left[0], y]
                    y_split_2 = [curr_patch.top_right[0], y]                
                                        
                    patch_1 = patch(top_left, top_right, y_split_2, y_split_1)
                    patch_2 = patch(y_split_1, y_split_2, bottom_right, bottom_left)
                    
                    curr.append(patch_1)
                    curr.append(patch_2)
        del curr [0:(self.patch_no-1)]
        self.all_patches = curr

class patched_img():
    def __init__(self, train_path, height, width, min_class, max_class, min_size):
        self.train_path = train_path
        self.height = height
        self.width = width
        self.min_class = min_class
        self.max_class = max_class
        self.min_size = min_size
        
    def generate_patched_img(self):
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
        
        imgs_count = random.randint(self.min_class, self.max_class)
    
        classes = os.listdir(self.train_path)
        
        selected = random.sample(classes, imgs_count)
        imgs = []
        for i in selected:
            class_path = os.path.join(self.train_path, i)
            img_path = os.path.join(class_path, random.choice(os.listdir(class_path)))
            img = cv2.imread(img_path)
            imgs.append(img)
            
        patch_list = patches(self.height, self.width, imgs_count, self.min_size)
        
        canvas = np.full((self.height, self.width, 3), (255, 255, 255), dtype=np.uint8)
        for i in range(0, imgs_count):
            patch = fit_to_patch(imgs[i], patch_list.all_patches[i])
            x = patch_list.all_patches[i].top_left[0]
            y =  patch_list.all_patches[i].top_left[1]
            patch_height, patch_width, _ = patch.shape
            canvas[y:y+patch_height, x:x+patch_width] = patch
            
        return canvas, selected

