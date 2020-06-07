import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from evalutation import Evaluation
from util import intersect_over_union, parse_xml, precision_recall_class

model_path = '/home/tomislav/Tomo/Faks/Zavrsni-rad/implementacija/object_detector/assets/ssd-tflite-model.tflite'
label_file = '/home/tomislav/Tomo/Faks/Zavrsni-rad/implementacija/object_detector/assets/pet_label_list.txt'
annotation_file = '/home/tomislav/Tomo/Faks/Zavrsni-rad/testing/extract_annotations'
input_mean = 127.5
input_std = 127.5

images_dir = 'images'

score_thres = 0.75


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() != '???']


def calc_precision(counter):
    precisions = {}
    for class_name, evaluation in counter.items():
        precisions[class_name] = evaluation.precision()
    return precisions


def retrieve_cords(rect, width, height):
    y_min = int(max(1, (rect[0] * height)))
    x_min = int(max(1, (rect[1] * width)))
    y_max = int(min(height, (rect[2] * height)))
    x_max = int(min(width, (rect[3] * width)))
    return [x_min, y_min, x_max, y_max]


def test_data(interpreter):
    pass


def classification_acc(result, evaluation, class_name):
    labels = load_labels(label_file)
    for index, score in enumerate(result['scores'][0]):
        if score > score_thres:
            cl = int(result['classes'][0][index])
            evaluation.add_true() if class_name == labels[cl] else evaluation.add_false()


def main():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f'Output details length: {len(output_details)}')

    floating_model = input_details[0]['dtype'] == np.float32

    dimensions = {
        'height': input_details[0]['shape'][1],
        'width': input_details[0]['shape'][2]
    }
    counter = {}

    for image_dir in os.listdir(images_dir):
        evaluation = Evaluation()
        counter[image_dir] = evaluation
        class_name = image_dir
        print(class_name)
        image_dir = os.path.join(images_dir, image_dir)

        positive_classes = []
        positive_scores = []

        for image in os.listdir(image_dir):

            if not image.endswith('.jpg') and not image.endswith('.png'):
                continue

            # image_xml_name = image.split('.')[0] + '.xml'
            # xml_attrs = parse_xml(os.path.join(annotation_file, class_name + '/' + image_xml_name))

            # if not xml_attrs:
            #     continue

            image = os.path.join(image_dir, image)
            img = Image.open(image)
            img = img.resize((dimensions['width'], dimensions['height']))

            if img.mode == 'RGBA':
                img.mode = 'RGB'

            input_data = np.expand_dims(img, axis=0)

            # img = cv2.imread(image)
            # img = cv2.resize(img, (dimensions['width'], dimensions['height']))

            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            result = {
                'rectangles': interpreter.get_tensor(output_details[0]['index']),
                'classes': interpreter.get_tensor(output_details[1]['index']),
                'scores': interpreter.get_tensor(output_details[2]['index']),
                'total': interpreter.get_tensor(output_details[3]['index']),
            }

            classification_acc(result, evaluation, class_name)
        #     appending_classes = []
        #     appending_scores = []
        #     for index, rect in enumerate(result['rectangles'][0]):
        #         cords = retrieve_cords(rect, dimensions['width'], dimensions['height'])
        #         iou = intersect_over_union(xml_attrs, cords)
        #         if iou > 0.5:
        #             appending_classes.append(result['classes'][0][index])
        #             appending_scores.append(result['scores'][0][index])
        #     if appending_classes:
        #         positive_classes.append(appending_classes)
        #     if appending_scores:
        #         positive_scores.append(appending_scores)
        # avg = precision_recall_class(y_true=positive_classes, y_scores=positive_scores)
        # print(f'AP for {class_name}: {avg}')
            #         draw_rect(img, cords, width=width, height=height)
            #         draw_rect(img, cords, width=width, height=height, color=(0, 0, 255))
            #         cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (255, 255, 255), 2)
            #         cv2.rectangle(img, (xml_attrs[0], xml_attrs[1]), (xml_attrs[2], xml_attrs[3]), (0, 0, 255), 2)
            #         cv2.imshow("image", img)
            #         cv2.waitKey(1)
            #     cl = int(classes[0][index])
            #     evaluation.add_true() if (
            #             class_name == labels[cl] and scores[0][index] > 0.5) else evaluation.add_false()

    print(counter)
    precisions = calc_precision(counter)
    print(f'{precisions}')

    res = {
        'counts': counter,
        'precisions': precisions
    }

    res_file = f'results_{score_thres}.pickle'

    with open(res_file, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
