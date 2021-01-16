import numpy as np
import tensorflow as tf
import imageio
import argparse
import os

from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets.util.hdf5_loader import load_data, load_qpos, default_loader_hparams, load_data_costumized

NUM_VISUAL_PER_VIEW = 3


def collect_same_viewpoint(robot, directory):
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [240, 320]

    meta_data = load_metadata(directory)

    for f in os.listdir(directory):
        if robot in f:
            path = directory + f
            print(path)
            imgs, states, qposes, ws_min, ws_max, viewpoint = load_data_costumized(
                path, meta_data.get_file_metadata(path), hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tests hdf5 data loader without tensorflow dataset wrapper")
    parser.add_argument('directory', type=str, help="path to dataset folder")
    parser.add_argument('robot', type=str, help="robot")
    args = parser.parse_args()

    collect_same_viewpoint(args.robot, args.directory)
