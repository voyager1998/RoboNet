# RoboNet

Code for loading and manipulating the RoboNet dataset, as well as for training supervised inverse models and video prediction models on the dataset.

Please refer to the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki) for more detailed documentation.

If you find the codebase or dataset useful please consider citing our paper.

```
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
```
