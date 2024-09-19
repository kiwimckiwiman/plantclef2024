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
    def __init__(self, height, width, min_size):
        self.height = height
        self.width = width
        self.min_size = min_size
        self.all_patches = []
        self.create_patches()
        
    def create_patches(self):
        init_box = patch([0,0], [self.width, 0], [self.width, self.height], [0, self.height])
        
        rand_point = [random.randint(self.min_size, self.width-self.min_size), random.randint(self.min_size, self.height-self.min_size)]
        
        tl = init_box.top_left
        tr = init_box.top_right
        br = init_box.bottom_right
        bl = init_box.bottom_left
        
        tm = [rand_point[0], tl[1]]
        rm = [tr[0], rand_point[1]]
        bm = [rand_point[0], bl[1]]
        lm = [tl[0], rand_point[1]]
        
        A = patch(tl, tm, rand_point, lm)
        B = patch(tm, tr, rm, rand_point)
        C = patch(lm, rand_point, bm, bl)
        D = patch(rand_point, rm, br, bm)
        
        self.all_patches = [A, B, C, D]
            
test = patches(500, 500, 100)

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