import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_detector():
    def __init__(self, img):
        self.img = img

    # @profile
    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.calc_gradient()
        hog_vector = []
        cell_gradient_vector = []

        for i in range(0, (height - (height % 16) - 1), 16):
            horizontal_vector = []
            for j in range(0, width - (width % 16) - 1, 16):
                cell_magnitude = gradient_magnitude[i:i + 16, j:j + 16]
                cell_angle = gradient_angle[i:i + 16, j:j + 16]
                horizontal_vector.append(self.calc_cell_gradient(cell_magnitude, cell_angle))
            cell_gradient_vector.append(horizontal_vector)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        height = len(cell_gradient_vector)
        width = len(cell_gradient_vector[0])

        for i in range(height - 1):
            for j in range(width - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def calc_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    # @profile
    def calc_cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = dict(
            [(20, 0), (60, 0), (100, 0), (140, 0), (180, 0), (220, 0), (260, 0), (300, 0), (340, 0)])
        # orientation_centers = dict(
        #     [(0, 0), (40, 0), (80, 0), (120, 0), (160, 0), (200, 0), (240, 0), (280, 0), (320, 0)])
        x_left = 0
        y_left = 0
        cell_magnitude = abs(cell_magnitude)
        for i in range(x_left, x_left + 16):
            for j in range(y_left, y_left + 16):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle = self.get_closest_bins(gradient_angle, orientation_centers)
                if min_angle == max_angle:
                    orientation_centers[min_angle] += gradient_strength
                else:
                    orientation_centers[min_angle] += gradient_strength * (abs(gradient_angle - max_angle) / 40)
                    orientation_centers[max_angle] += gradient_strength * (abs(gradient_angle - min_angle) / 40)
        return orientation_centers.values()

    def get_closest_bins(self, gradient_angle, orientation_centers):
        angles = []
        for angle in orientation_centers:
            if abs(gradient_angle - angle) < 40:
                angles.append(angle)
        angles.sort()
        if len(angles) == 1:
            return angles[0], angles[0]
        return angles[0], angles[1]

    def render_gradient(self, image, cell_gradient):
        height, width = image.shape
        height = height - (height % 16) - 1
        width = width - (width % 16) - 1
        x_start = 8
        y_start = 8
        cell_width = 8
        max_mag = np.array(cell_gradient).max()
        for x in range(x_start, height, 16):
            for y in range(y_start, width, 16):
                cell_x = x / 16
                cell_y = y / 16
                cell_grad = cell_gradient[cell_x][cell_y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = 40
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


img = cv2.imread('person_037.png', cv2.IMREAD_GRAYSCALE)
hog = Hog_detector(img)
vector, image = hog.extract()
print np.array(vector).shape
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
