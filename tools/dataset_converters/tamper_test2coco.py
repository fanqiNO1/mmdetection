import argparse
import os

import imagesize
from mmengine.fileio import dump
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert tamper dataset to COCO format')
    parser.add_argument('src', help='source dataset path')
    args = parser.parse_args()
    return args


def collect_image_infos(src):
    image_paths = []
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith('jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    image_infos = []
    for image_path in tqdm(image_paths):
        # get relative path
        filename = os.path.relpath(image_path, src)
        img_w, img_h = imagesize.get(image_path)
        image_infos.append({
            'filename': filename,
            'width': img_w,
            'height': img_h
        })
    return image_infos


def cvt_to_coco_json(image_infos):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []

    # category
    category = dict()
    category['supercategory'] = 'none'
    category['id'] = 0
    category['name'] = 'tampered'
    coco['categories'].append(category)

    # images
    for image_info in image_infos:
        # image
        filename = image_info['filename']
        image_item = dict()
        image_item['id'] = image_id
        image_item['file_name'] = filename
        image_item['height'] = image_info['height']
        image_item['width'] = image_info['width']
        coco['images'].append(image_item)

        image_id += 1
    return coco


def main():
    args = parse_args()
    image_infos = collect_image_infos(args.src)
    coco = cvt_to_coco_json(image_infos)
    dump(coco, os.path.join(args.src, 'tamper.json'))


if __name__ == '__main__':
    main()
