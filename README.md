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
python dataset_visualizer.py hdf5 --robots baxter
python dataset_visualizer.py hdf5 --robots baxter --batch_size 8
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

Penn uses Logitech C920 camera, and the camera intrinsic is `[641.5, 0, 320.0, 0, 641.5, 240.0, 0, 0, 1]`.

## Workspace Dimension

```text
"baxter_right": [[ 0.40, -0.67 ,  -0.15, 15.0,  0.0 ], [ 0.75, -0.20,  -0.05,  3.2e+02,  1.0e+02]],
"baxter_left": [[ 0.45,  0.15, -0.15, 15.0,  0.0 ], [  0.75,  0.59, -0.05,  3.2e+02,  1.0e+02]],
"vestri": [[0.47, -0.2, 0.176, 1.5707963267948966, -1], [0.81, 0.2, 0.292, 4.625122517784973, 1]],
"vestri_table": [[0.43, -0.34, 0.176, 1.5707963267948966, -1], [0.89, 0.32, 0.292, 4.625122517784973, 1]],
"vestri_table_default": [[0.43, -0.34, 0.176, 1.5707963267948966, -1], [0.89, 0.32, 0.292, 4.625122517784973, 1]],
"sudri": [[0.45, -0.18, 0.176, 1.5707963267948966, -1], [0.79, 0.22, 0.292, 4.625122517784973, 1]],
"test": [[0.47, -0.2, 0.1587, 1.5707963267948966, -1], [0.81, 0.2, 0.2747, 4.625122517784973, 1]],
"baxter": [[ 0.6, -0.5 , -0.18, 15.0,  0.0 ], [ 8.0e-01, -5.0e-02,  -0.0316,  3.2e+02,  1.0e+02]]
```
