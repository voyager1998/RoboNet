# RoboNet

Code for loading and manipulating the RoboNet dataset, as well as for training supervised inverse models and video prediction models on the dataset.

Please refer to the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki) for more detailed documentation.

If you find the codebase or dataset useful please consider citing our paper.

```text
@inproceedings{dasari2019robonet,
    title={RoboNet: Large-Scale Multi-Robot Learning},
    author={Sudeep Dasari and Frederik Ebert and Stephen Tian and Suraj Nair and Bernadette Bucher and Karl Schmeckpeper and Siddharth Singh and Sergey Levine and Chelsea Finn},
    year={2019},
    eprint={1910.11215},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    booktitle={CoRL 2019: Volume 100 Proceedings of Machine Learning Research}
}
```

## Downloading the Dataset

You can find instructions for downloading the dataset on the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki/Getting-Started) as well. All data is provided under the [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

## Dataset Visualization

```bash
python dataset_visualizer.py hdf5
python dataset_visualizer.py hdf5 --robots widowx
```

Candidate Robot Names: sawyer, kuka, R3, widowx, baxter, fetch, franka

## Dataset Spec

State: 5D, [(x, y, z, rotation)?, gripper]

## Robot Self Recognition

* Robot-Supervised Learning for Object Segmentation - requires depth images
* Self-Supervised Object-in-Gripper Segmentation from Robotic Motions - not open-source

## Camera Calibration

1. `mkdir camera_calib/mujoco_gts`
2. Copy all `hdf5` data under `camera_calib/mujoco_gts`
3. Run `python camera_calib/mujoco_gt_test.py`
