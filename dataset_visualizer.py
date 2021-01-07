import numpy as np
import tensorflow as tf
import imageio
import argparse

from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--robots', type=str, nargs='+', default=None,
                        help='will construct a dataset with batches split across given robots')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'RNG': 0,
               'ret_fnames': True,
               #    'load_annotations': True, # does not contain annotations?
               'load_T': args.load_steps,
               'sub_batch_size': 8,
               'action_mismatch': 3,
               'state_mismatch': 3,
               'splits': [0.8, 0.1, 0.1],
               'same_cam_across_sub_batch': True,
               'img_size': [240, 320]}

    if args.robots:
        meta_data = load_metadata(args.path)
        hparams['same_cam_across_sub_batch'] = True
        loader = RoboNetDataset(args.batch_size, [meta_data[meta_data['robot'] == r]
                                                  for r in args.robots], hparams=hparams)
    else:
        loader = RoboNetDataset(args.batch_size, args.path, hparams=hparams)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'f_names']]
    s = tf.Session()
    out_tensors = s.run(tensors, feed_dict=loader.build_feed_dict(args.mode))

    imgs = out_tensors[0]
    print("image shape", imgs.shape)
    states = out_tensors[1]
    print("state shape", states.shape)
    actions = out_tensors[2]
    print("action shape", actions.shape)

    writer = imageio.get_writer('images/test_frames.gif')
    for t in range(imgs.shape[1]):
        imageio.imwrite("images/test_imgs_" + str(t) + ".png", (imgs[0, t, 0] * 255).astype(np.uint8))
        print("state:   ", states[0, t])
        # if t < imgs.shape[1]-1:
        # print("action:  ", actions[0, t])
        for i in range(1):
            writer.append_data((imgs[0, t, 0] * 255).astype(np.uint8))
    writer.close()
