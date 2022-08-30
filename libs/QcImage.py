import numpy as np


RECT_SCALE = 1000


def get_average_rgb(image_data):
    return np.average(image_data, axis=(0, 1))


def crop_image_by_position_and_rect(cv_image, position, rect):
    # img[y: y + h, x: x + w]
    height = cv_image.shape[0]
    width = cv_image.shape[1]
    position_x = position.x * width
    position_y = position.y * height
    rect_x = width * rect.x / RECT_SCALE
    rect_y = height * rect.y / RECT_SCALE
    return cv_image[int(position_y):int(position_y) + int(rect_y),
                    int(position_x):int(position_x) + int(rect_x)]
