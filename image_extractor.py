import os
import re
from shutil import copy

root = '/home/tomislav/Tomo/Faks/Zavrsni-rad/testing/'

test_file = '/home/tomislav/Tomo/Faks/Zavrsni-rad/testing/test.txt'
annotation_dir = '/home/tomislav/Tomo/Faks/Zavrsni-rad/testing/xmls'
src_images_dir = '/home/tomislav/Tomo/Faks/Zavrsni-rad/temp/images'
dest_images_dir = '/home/tomislav/Tomo/Faks/Zavrsni-rad/testing/images'


def main():
    pattern = '^.+?(?=_\\d)'
    classes_to_extract = ['Abyssinian', 'basset_hound', 'beagle', 'chihuahua', 'Russian_Blue', 'Sphynx',
                          'staffordshire_bull_terrier']

    dir_sizes = {class_name: 0 for class_name in classes_to_extract}
    dir_size_limit = 100

    annotations = os.listdir(annotation_dir)
    src_images = os.listdir(src_images_dir)
    extracted_annotations = root + 'extract_annotations'

    if not os.path.exists(extracted_annotations):
        os.mkdir(extracted_annotations)
    with open(test_file, mode='r', encoding='utf-8') as test:
        class_name = test.readline().strip()
        name_match = re.match(pattern=pattern, string=class_name).group(0)
        print(name_match)
        while class_name.startswith(name_match):
            if class_name == '':
                continue
            class_name_with_number = class_name.split()[0]
            annotation_file_name = class_name_with_number + '.xml'
            print(f'File {annotation_file_name} in annotations: {annotation_file_name in annotations}')
            if annotation_file_name in annotations:
                annotation_file = os.path.join(annotation_dir, annotation_file_name)
                copy(annotation_file, os.path.join(extracted_annotations, annotation_file_name))

            image_file_jpg_name = class_name_with_number + '.jpg'
            if image_file_jpg_name in src_images:
                class_image_dest_dir = os.path.join(dest_images_dir, name_match)
                if not os.path.exists(class_image_dest_dir):
                    os.mkdir(class_image_dest_dir)
                if dir_sizes[name_match] < dir_size_limit:
                    image_file = os.path.join(src_images_dir, image_file_jpg_name)
                    dest_image = os.path.join(class_image_dest_dir, image_file_jpg_name)
                    if not os.path.exists(dest_image):
                        copy(image_file, dest_image)
                    dir_sizes[name_match] += 1

            class_name = test.readline().strip()
            if not class_name.startswith(name_match):
                class_name = test.readline().strip()
                name_match = re.match(pattern=pattern, string=class_name)
                if not name_match:
                    break
                name_match = name_match.group(0)
                while name_match not in classes_to_extract:
                    class_name = test.readline().strip()
                    if class_name.startswith(name_match):
                        continue
                    name_match = re.match(pattern=pattern, string=class_name)
                    if not name_match:
                        break
                    name_match = name_match.group(0)
            if not name_match:
                break


if __name__ == '__main__':
    main()
