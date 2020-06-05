import xml.etree.ElementTree as et

import cv2
from sklearn.metrics import average_precision_score


def parse_xml(xml_file):
    tree = et.parse(xml_file)
    root = tree.getroot()
    bndbox = root.find('object/bndbox')

    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def draw_rect(image, box, height, width, color=(255, 255, 255), thickness=2):
    y_min = box[1]
    x_min = box[0]
    y_max = box[3]
    x_max = box[2]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


