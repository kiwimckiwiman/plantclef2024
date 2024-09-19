import random
import tkinter as tk

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

test = patches(500, 500, random.randint(2,6), 100)

def rgbtohex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

window = tk.Tk()
window.geometry("600x600")
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
window.mainloop()