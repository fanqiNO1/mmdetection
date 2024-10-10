import argparse
import json
import os

import imagesize
import numpy as np
import pandas as pd
from mmengine.fileio import dump
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert tamper dataset to COCO format')
    parser.add_argument('src', help='source dataset path')
    parser.add_argument('ann', help='annotation file path')
    args = parser.parse_args()
    return args


def prepare_anno(src, ann):
    data = pd.read_csv(ann)
    anns = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        img_path = row['Path']
        img_w, img_h = imagesize.get(os.path.join(src, img_path))
        polygons = json.loads(row['Polygons'])
        bboxes = [np.array(x, dtype=np.int64) for x in polygons]
        bboxes = np.array([(bbox[:, 0].min(), bbox[:, 1].min(),
                            bbox[:, 0].max(), bbox[:, 1].max())
                           for bbox in bboxes],
                          dtype=np.int64)
        anns.append({
            'filename': img_path,
            'width': img_w,
            'height': img_h,
            'bboxes': bboxes
        })
    return anns


def cvt_to_coco_json(anns):
    image_id = 0
    id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []

    # category
    category = dict()
    category['supercategory'] = 'none'
    category['id'] = 0
    category['name'] = 'tampered'
    coco['categories'].append(category)

    # images
    for ann in anns:
        # image
        file_name = ann['filename']
        image_item = dict()
        image_item['id'] = image_id
        image_item['file_name'] = file_name
        image_item['height'] = ann['height']
        image_item['width'] = ann['width']
        coco['images'].append(image_item)
        # annotation
        bboxes = ann['bboxes']  # xmin, ymin, xmax, ymax
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            ann_item = dict()
            ann_item['segmentation'] = [
                x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min
            ],
            ann_item['area'] = int((x_max - x_min) * (y_max - y_min))
            ann_item['ignore'] = 0
            ann_item['iscrowd'] = 0
            ann_item['image_id'] = image_id
            ann_item['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
            ann_item['category_id'] = 0
            ann_item['id'] = id
            id += 1
            coco['annotations'].append(ann_item)
        image_id += 1
    return coco


def main():
    args = parse_args()
    anns = prepare_anno(args.src, args.ann)
    coco = cvt_to_coco_json(anns)
    dump(coco, os.path.join(args.src, 'tamper.json'))


if __name__ == '__main__':
    main()
