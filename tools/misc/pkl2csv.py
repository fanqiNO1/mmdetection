import argparse
import os
import pickle

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert pkl results to csv format')
    parser.add_argument('src', help='source pkl file')
    parser.add_argument('dst', help='destination csv file')
    parser.add_argument('--data_root', help='data root')
    parser.add_argument('--score_thr', type=float, default=0.5)
    args = parser.parse_args()
    return args


def convert_to_csv(data, data_root, score_thr):
    csv_data = dict(Path=[], Polygon=[])
    for i in tqdm(range(len(data))):
        path = data[i]['img_path']
        path = os.path.relpath(path, data_root)
        bboxes = data[i]['pred_instances']['bboxes']
        scores = data[i]['pred_instances']['scores']
        polygons = []
        for box, score in zip(bboxes, scores):
            if score < score_thr:
                continue
            x1, y1, x2, y2 = box.tolist()
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            polygons.append(polygon)
        csv_data['Path'].append(path)
        csv_data['Polygon'].append(polygons)
    return csv_data


def main():
    args = parse_args()
    with open(args.src, 'rb') as f:
        data = pickle.load(f)
    data = convert_to_csv(data, args.data_root, args.score_thr)
    df = pd.DataFrame(data)
    df.to_csv(args.dst, index=False)


if __name__ == '__main__':
    main()
