import os
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class AnnotationTool:
    def __init__(self, image_folder, output_folder):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.images = sorted(os.listdir(image_folder))
        self.current_index = 0
        self.plant_count = 0
        self.nonplant_count = 0

        self.window = tk.Tk()
        self.window.title("Image Annotation Tool")

        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        self.plant_button = tk.Button(self.button_frame, text="Plant", command=lambda: self.annotate_image("plant"))
        self.plant_button.pack(side="left")

        self.nonplant_button = tk.Button(self.button_frame, text="Non-Plant", command=lambda: self.annotate_image("nonplant"))
        self.nonplant_button.pack(side="left")

        self.status_label = tk.Label(self.window, text="")
        self.status_label.pack()

        self.update_image()

        self.window.mainloop()

    def update_image(self):
        if self.current_index < len(self.images):
            image_path = os.path.join(self.image_folder, self.images[self.current_index])
            img = Image.open(image_path)
            img.thumbnail((500, 500))  # Adjust the size as needed
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.status_label.config(text=f"Image {self.current_index + 1} of {len(self.images)} - "
                                          f"Plants: {self.plant_count}, Non-Plants: {self.nonplant_count}")
        else:
            messagebox.showinfo("Completed", "All images have been annotated.")
            self.window.quit()

    def annotate_image(self, label):
        if self.current_index < len(self.images):
            image_name = self.images[self.current_index]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{label}_{timestamp}_"
            src_path = os.path.join(self.image_folder, image_name)
            dest_path = os.path.join(self.output_folder, new_name)

            shutil.copy(src_path, dest_path)

            if label == "plant":
                self.plant_count += 1
            else:
                self.nonplant_count += 1

            self.current_index += 1
            self.update_image()

if __name__ == "__main__":
    image_folder = r"D:\plantclef2024\p_np_set"  # Change this to your image folder
    output_folder = r"D:\plantclef2024\p_np"   # Change this to your output folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tool = AnnotationTool(image_folder, output_folder)
