import numpy as np
import tensorflow as tf
import imageio
import argparse
import os

from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets.util.hdf5_loader import load_data, load_qpos, default_loader_hparams, load_data_customized

NUM_VISUAL_PER_VIEW = 3


def collect_same_viewpoint(robot, directory):
    hparams = tf.contrib.training.HParams(**default_loader_hparams())

    meta_data = load_metadata(directory)

    exp_same_view = {}

    for f in os.listdir(directory):
        if robot in f:
            path = directory + f
            print(path)
            _, _, _, _, _, viewpoint = load_data_customized(
                path, meta_data.get_file_metadata(path), hparams)
            if viewpoint not in exp_same_view:
                exp_same_view[viewpoint] = [path]
            else:
                exp_same_view[viewpoint].append(path)
    return exp_same_view


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tests hdf5 data loader without tensorflow dataset wrapper")
    parser.add_argument('directory', type=str, help="path to dataset folder")
    parser.add_argument('robot', type=str, help="robot")
    args = parser.parse_args()

    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [240, 320]

    meta_data = load_metadata(args.directory)

    exp_same_view = collect_same_viewpoint(args.robot, args.directory)
    print(len(exp_same_view))

    os.makedirs(args.robot, exist_ok=True)
    for vp in exp_same_view:
        target_folder = args.robot + '/' + vp
        os.makedirs(target_folder, exist_ok=True)
        visuals = min(NUM_VISUAL_PER_VIEW, len(exp_same_view[vp]))
        for i in range(visuals):
            f = exp_same_view[vp][i]
            exp_name = f.split('/')[-1][:-5]
            imgs, states, qposes, ws_min, ws_max, viewpoint = load_data_customized(f,
                                                                                   meta_data.get_file_metadata(f),
                                                                                   hparams)
            print("saving experiment:", exp_name)
            np.save(target_folder + "/states_" + exp_name, states)
            np.save(target_folder + "/qposes_" + exp_name, qposes)

            writer = imageio.get_writer(target_folder + '/' + exp_name + '.gif')
            for t in range(imgs.shape[0]):
                for i in range(1):
                    writer.append_data(imgs[t, 0])
            writer.close()
