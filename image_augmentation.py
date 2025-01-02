import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image


class ImageAugmentation:
    """
    Image augmentation class
    """
    def __init__(self, image, label):
        self.image = image
        self.label = label
        
    def rotate(self, angle):
        """
        Rotate the image by a random angle between -angle and angle
        """
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta) 
        
        height, width = self.image.shape
        center_x, center_y = width // 2, height // 2
        
        y, x = np.mgrid[0:height, 0:width]
        x = x - center_x
        y = y - center_y
        
        x_rot = x * cos_theta - y * sin_theta + center_x
        y_rot = x * sin_theta + y * cos_theta + center_y
        
        # Interpolate values
        x_rot = np.clip(x_rot, 0, width-1)
        y_rot = np.clip(y_rot, 0, height-1)
        
        rotated = np.zeros_like(self.image)
        for i in range(height):
            for j in range(width):
                x_new, y_new = int(x_rot[i,j]), int(y_rot[i,j])
                rotated[i,j] = self.image[y_new, x_new]
                
        return rotated
    
    def zoom(self, factor):
        """
        Zoom the image by a factor, it could be zoom in or zoom out
        """
        height, width = self.image.shape
        new_height = int(height * factor)
        new_width = int(width * factor)
        
        # Calculate padding/cropping
        y_start = (new_height - height) // 2
        x_start = (new_width - width) // 2
        
        zoomed = np.zeros((new_height, new_width))
        
        # if factor > 1, zoom in
        if factor > 1:
            zoomed[y_start:y_start+height, x_start:x_start+width] = self.image
            # Crop back to original size
            y_crop = (new_height - height) // 2
            x_crop = (new_width - width) // 2
            zoomed = zoomed[y_crop:y_crop+height, x_crop:x_crop+width]
        else:  # Zoom out
            zoomed = self.image[-y_start:height+y_start, -x_start:width+x_start]
            # Resize back to original size
            zoomed = ndimage.zoom(zoomed, (height/zoomed.shape[0], width/zoomed.shape[1]))
        
        return zoomed
    
    def add_blur(self, kernel_size=3):
        """
        Add Gaussian blur to the image
        """
        return gaussian_filter(self.image, sigma=kernel_size/3)
    
    def augment(self, angle=15, factor=0.5, kernel_size=3):
        """
        Augment the image by rotating, zooming, and adding blur
        and return the augmented image and label
        """
        random_angles = np.random.randint(-angle, angle)
        random_factors = np.random.uniform(1-factor, 1+factor)
        random_kernel_sizes = np.random.randint(0.4, kernel_size)
        
        aug_img = self.rotate(random_angles)
        aug_img = self.zoom(random_factors)
        aug_img = self.add_blur(random_kernel_sizes)
    
        
        return aug_img, self.label

    def remove_pixels_randomly(self, percentage):
        """
        Remove pixels randomly from the image
        """
        width, height = self.image.shape
        num_pixels_to_remove = int(width * height * percentage)
        for _ in range(num_pixels_to_remove):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            self.image[y, x] = 255
        return self.image, self.label
    
    def remove_pixels_square(self, percentage):
        """
        Remove pixels from the image in a square shape
        """
        width, height = self.image.shape
        square = int(width * percentage)
        if percentage > 0.3:
            print('Attention : removing is too much')
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        self.image[y:min(y+square, height), x:min(x+square, width)] = 0
        return self.image, self.label

if __name__ == "__main__":
    #image = np.random.randint(0, 255, (100, 100)) / 255
    image = np.ones((100, 100)) * 255  # Multiply by 255 to get white pixels
    pil_image = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 for PIL
    pil_image.show()
    label = 5
    augmenter = ImageAugmentation(image, label)
    augmented_images, augmented_labels = augmenter.augment(4)
    augmented_images = augmenter.remove_pixels_square(0.4)
    pil_image = Image.fromarray(augmented_images)
    pil_image.show()
    