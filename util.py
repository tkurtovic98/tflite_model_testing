import os
import xml.etree.ElementTree as et

import cv2
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer


def parse_xml(xml_file):
    if not os.path.exists(xml_file):
        return []
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


"""
Code for iou taken from http://ronny.rest/tutorials/module/localization_001/iou/
"""


def intersect_over_union(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
               [x1,y1,x2,y2]
           where:
               x1,y1 represent the upper left corner
               x2,y2 represent the lower right corner
           It returns the Intersect of Union score for these two boxes.

       Args:
           a:          (list of 4 numbers) [x1,y1,x2,y2]
           b:          (list of 4 numbers) [x1,y1,x2,y2]
           epsilon:    (float) Small value to prevent division by zero

       Returns:
           (float) The Intersect of Union score.
       """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def precision_recall_class(y_true, y_scores):
    precision = dict()
    recall = dict()
    average_precision = dict()
    classes = [i for i in range(0, 37)]

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true)

    y_true = label_binarize([[int(y) for y in sub_list] for sub_list in y_true], classes=classes)

    for i in list(mlb.classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_scores[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_scores[:, i])
