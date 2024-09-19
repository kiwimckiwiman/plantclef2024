import tkinter as tk
import random

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
    
    def add_patches(self, selected, box1, box2):
        self.all_patches.remove(selected)
        self.all_patches.append(box1)
        self.all_patches.append(box2)
        
    def create_patches(self):
        init_box = patch([0,0], [self.width, 0], [self.width, self.height], [0, self.height])
        curr = []
        curr.append(init_box)
        
        def split_box(patches, max_patches, min_size, x_or_y):
            if max_patches == 1:
                return patches
            else:
                random.shuffle(patches)
                for box in patches:
                    if(x_or_y == 1 and box.width() >= min_size*2):#odd = fixed x point
                        num_sections = box.width()/min_size
                        x = (random.randint(1, num_sections-1))*min_size
                        
                        top_left = box.top_left
                        top_right = box.top_right
                        bottom_right = box.bottom_right
                        bottom_left = box.bottom_left
                        
                        x_split_1 = [x, box.top_left[1]]
                        x_split_2 = [x, box.bottom_left[1]]                      
                                            
                        box_1 = patch(top_left, x_split_1, x_split_2, bottom_left)
                        box_2 = patch(x_split_1, top_right, bottom_right, x_split_2)
                        
                        curr = []
                
                        patches.remove(box)
                        curr = patches
                        curr.append(box_1)
                        curr.append(box_2)
                        
                        return split_box(curr, max_patches-1, min_size, ((x_or_y + 1)%2))
                        break
                    elif(x_or_y == 0 and box.height() >= min_size*2):#even = fixed y point
                        num_sections = box.height()/min_size
                        y = (random.randint(1, num_sections-1))*min_size
                                       
                        top_left = box.top_left
                        top_right = box.top_right
                        bottom_right = box.bottom_right
                        bottom_left = box.bottom_left
                        
                        y_split_1 = [box.top_left[0], y]
                        y_split_2 = [box.top_right[0], y]                
                                            
                        box_1 = patch(top_left, top_right, y_split_2, y_split_1)
                        box_2 = patch(y_split_1, y_split_2, bottom_right, bottom_left)
                        
                        curr = []
                        
                        patches.remove(box)
                        curr = patches
                        curr.append(box_1)
                        curr.append(box_2)
                        
                        return split_box(curr, max_patches-1, min_size, ((x_or_y + 1)%2))
                        break
                    
        self.all_patches = split_box(curr, self.patch_no, self.min_size, (random.randint(0,100)%2))
            
test = patches(500, 500, 4, 100)

window = tk.Tk()
def rgbtohex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

for i in test.all_patches:
    color = rgbtohex(random.randint(0,255), random.randint(0,255), random.randint(0,255))
    frame1 = tk.Frame(master=window, width=i.width(), height=i.height(), bg=str(color))
    frame1.place(x=i.top_left[0], y=i.top_left[1])
    label1 = tk.Label(master=window, text=str(i.top_left), bg=str(color))
    label1.place(x=i.top_left[0], y=i.top_left[1])
    
    label2 = tk.Label(master=window, text=str(i.top_right), bg=str(color))
    label2.place(x=i.top_right[0]-60, y=i.top_right[1])
    
    label3 = tk.Label(master=window, text=str(i.bottom_left), bg=str(color))
    label3.place(x=i.bottom_left[0], y=i.bottom_left[1]-20)
    
    label4 = tk.Label(master=window, text=str(i.bottom_right), bg=str(color))
    label4.place(x=i.bottom_right[0]-60, y=i.bottom_right[1]-20)
    i.coords()
window.mainloop()

                    #                    x_split_1
                    #top_left     .----------.-----------.   top_right
                    #             |          |           |
                    #             |          |           |
                    #             |          |           |
                    #             |          |           |
                    #             |          |           |
                    #             |          |           |
                    #bottom_left  .----------.-----------.   bottom_right
                    #                    x_split_2

                    #top_left     .----------------------.   top_right
                    #             |                      |
                    #             |                      |
                    #             |                      |
                    #y_split_1    .----------------------.   y_split_2
                    #             |                      |
                    #             |                      |
                    #             |                      |
                    #bottom_left  .----------------------.   bottom_right
