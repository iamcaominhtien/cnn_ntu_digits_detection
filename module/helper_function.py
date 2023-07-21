import time
import cv2
import numpy as np
import datetime, os
import math
from skimage.feature import hog

SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


def preprocess(image, level):
    # sharpen image
    image = cv2.filter2D(image, -1, SHARPEN_KERNEL)

    # convert to gray image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth image
    blurred = cv2.GaussianBlur(gray_img, (level, level), 0)  # Lam min anh

    # threshold image to binary image
    # threshold = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    im_bw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15,
                                  8)  # Nhi phan anh

    return im_bw


def length_of_line(line):
    x1, y1, x2, y2 = line
    return np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2))


def imshow(img, window_name='Image.jpg', width_size=None):
    if width_size is None:
        cv2.imshow(window_name, img)
    else:
        h_raw, w_raw, *_ = img.shape
        cv2.imshow(window_name, cv2.resize(img, (width_size, int(width_size * h_raw / w_raw))))


def find_intersection_of_2_lines(line1, line2):
    # convert to float
    x1, y1, x2, y2 = [i * 1.0 for i in line1]
    x3, y3, x4, y4 = [i * 1.0 for i in line2]

    if (x1 == x2 and y1 == y2) or (x3 == x4 and y3 == y4):
        return None

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return [int(x), int(y)]


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def analyze_data_to_find_outliers(data):
    """
      This function analyzes a dataset to find outliers.

      Args:
        data: A list or NumPy array of data points.

      Returns:
        A dictionary with the following keys:
          outlier_index: A list of indices of the outliers.
          clean_index: A list of indices of the non-outliers.
          median: The median of the data.
      """
    if len(data) > 0:
        data = np.array(data)
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        return {
            'outlier_index': np.where((data > upper_bound) | (data < lower_bound))[0],
            'clean_index': np.where((data <= upper_bound) & (data >= lower_bound))[0],
            'median': q2,
        }
    else:
        return {
            'outlier_index': [],
            'clean_index': [],
            'median': None,
        }


def line_slope_intercept(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        slope = None
        intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    return slope, intercept


def save_img(img, path_dir, folders=None, name=None, tail='.png'):
    if folders is None:
        folders = []
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    if len(folders) > 0:
        for folder in folders:
            path_dir = os.path.join(path_dir, folder)
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)
    if name is None:
        name = current_time
    else:
        name = name + '_' + current_time + tail
    cv2.imwrite(os.path.join(path_dir, name), img)


# remove all file in folder, include subfolder
def clear_all_file(path_dir, included_sub_folder=True):
    import shutil

    for root, dirs, files in os.walk(path_dir):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))
            if not included_sub_folder:
                os.makedirs(os.path.join(root, dir))


def get_angle(line, rad=True):
    x1, y1, x2, y2 = line
    if x2 - x1 != 0:
        angle = math.atan2(y2 - y1, x2 - x1)
        if rad:
            return angle
        return math.degrees(angle)
    else:
        return None


def crop_ver_and_hor_lines(vers, hors):
    """
      This function crops a set of vertical and horizontal lines to the intersection of the lines with the top, bottom, left, and right edges of the image.

      Args:
        vers: A list of tuples representing vertical lines. Each tuple has the form (x1, y1, x2, y2).
        hors: A list of tuples representing horizontal lines. Each tuple has the form (x1, y1, x2, y2).

      Returns:
        A tuple of two lists. The first list contains the cropped vertical lines, and the second list contains the cropped horizontal lines.
      """
    top, bottom = hors[0], hors[-1]
    left, right = vers[0], vers[-1]

    cropped_ver_lines = []
    for line in vers:
        x1, y1 = find_intersection_of_2_lines(line, top)
        x2, y2 = find_intersection_of_2_lines(line, bottom)
        cropped_ver_lines.append([x1, y1, x2, y2])

    cropped_hor_lines = []
    for line in hors:
        x1, y1 = find_intersection_of_2_lines(line, left)
        x2, y2 = find_intersection_of_2_lines(line, right)
        cropped_hor_lines.append([x1, y1, x2, y2])

    return np.array(cropped_ver_lines).astype(int), np.array(cropped_hor_lines).astype(int)


def hog_feature_extraction(images, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)):
    """
      This function extracts Histogram of Oriented Gradients (HOG) features from a list of images.

      Args:
        images: A list of images.
        orientations: The number of orientations to use in the HOG feature descriptor.
        pixels_per_cell: The size of each cell in the HOG feature descriptor.
        cells_per_block: The number of cells in each block in the HOG feature descriptor.

      Returns:
        A list of HOG feature vectors.
      """
    images_hog = []

    for image in images:
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))
        fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, visualize=False)
        images_hog.append(fd)

    return np.array(images_hog)
