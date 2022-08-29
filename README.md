# Learning to Use Chopsticks in Diverse Gripping Styles
[Zeshi Yang](https://github.com/zeshiYang),
[KangKang Yin](https://www.cs.sfu.ca/~kkyin/),
and [Libin Liu](http://libliu.info/)

ACM Trans. Graph., Vol. 41, No. 4, Article 95. (SIGGRAPH 2022)

[Paper](https://arxiv.org/abs/2205.14313)
[Video](https://www.youtube.com/watch?v=rQHzwnSdsP8)

---

## Table of Contents
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Running demo](#running-demo)
1. [Training](#training)

## Citation
```bibtex
@article{
author = {Yang, Zeshi and Yin, Kangkang and Liu, Libin},
title = {Learning to Use Chopsticks in Diverse Gripping Styles},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {4},
journal = {ACM Trans. Graph.},
articleno = {95},
numpages = {17}
}
```




## Requirements and Dependencies
- mujoco_py (version:1.50.1.1)
- PyGLM (version: 0.4.8b1)
- torch (version:1.9.0+cu11)
- pytorch3d (latest)
- Everything else in `requirements.txt`

## Installation

If you need help with installing the dependecies please refer to our installation guide in [installation.md](installation.md).

## Running demo
To run our demo, run
```
python test.py --xml ./released_models/demo.xml --traj ./released_models/demo.txt --model_path ./released_models/model.tar
```
to visualize our pre-trained control policy for some chopsticks grasping tasks.

## Training
To train policies with generated trajectories, run
```
python train.py --xml 'path_to_MJCF' --traj 'path_of_trajectories' --threads 'num_threads' --logdir 'path_to_save_models'
```

For example:

```
python train.py --xml ./data/hand_xml_grasp/demo.xml --traj ./data/traj/ --threads 16 --logdir ./results
```

You can modify the `task_placeholder.py` to generate your own training data.
Run
```
python task_placeholder.py --chop_xml ./data/xml_templates/standard/0/chop_kin.xml --hand_xml ./data/xml_templates/standard/0/sim.xml --task_name 'your_task_name' --pose ./data/xml_templates/standard/0/standard.txt
```
to generate trajectories and MJCF file used for training. The generated motion files will store in `./data/your_task_name/` and the MJCF file will be in `./data/hand_xml_grasp/`.




