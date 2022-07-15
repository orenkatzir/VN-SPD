# Shape-Pose Disentanglement using SE(3)-equivariant Vector Neurons

[[Project]](https://github.com/orenkatzir/VNT) [[Paper]](https://arxiv.org/pdf/2204.01159.pdf)

This repository includes implementations to the method presented in the paper "Shape-Pose Disentanglement using SE(3)-equivariant Vector Neurons", including additional support for losses types and architectures (see code).

## Data Preparation
We use the ShapeNet dataset as in AtlasNetV2. Please follow the instructions at [CanonicalCapsules](https://github.com/canonical-capsules/canonical-capsules) to convert the data to h5 files. 
Other datasets will be supported shortly.

## Installation
The Dockerfile and the requirements.txt file include all the required dependencies. Please build and run it.
```
git clone VNT 
cd VNT
docker build -f Dockerfile -t vnt --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
docker run -t -d --gpus all -v ${PWD}:/app/VNT --name vnt_experiment -p 8097:8097 vnt
```
Execute the running container:
```
docker exec -it vnt_experiment /bin/bash
cd VNT
```

## Usage

### Training
To view training results and loss plots, run 
```
python -m visdom.server
```
 and click the URL [http://localhost:8097/](http://localhost:8097/)

Once visdom is up, start training
```
python train.py --model shape_pose --name chairs_paper --dataroot datasets/shapenet --class_choice chair --add_noise --apply_can_rot_loss --remove_knn 100 --resample --fps
```

An improved version of defaults can be seen in shape_pose2 model:
```
python train.py --model shape_pose2 --name chairs_updated --dataroot datasets/shapenet --class_choice chair --add_noise --apply_can_rot_loss --remove_knn 100 --resample --fps
``` 

### Evaluation
```
python test.py --model shape_pose --name chairs_paper --dataroot datasets/shapenet --class_choice chair
```
or
```
python test.py --model shape_pose2 --name chairs_updated --dataroot datasets/shapenet --class_choice chair
```

### Pre-trained model can be found [here](https://drive.google.com/drive/folders/1yUgv0NOAF7BDEYQEOnEWNjiA0PgPgohr?usp=sharing)
Download the models and place the extracted folder under the checkpoints directory. 

## Citation

    @article{katzir2022shape,
      title={Shape-Pose Disentanglement using SE (3)-equivariant Vector Neurons},
      author={Katzir, Oren and Lischinski, Dani and Cohen-Or, Daniel},
      journal={arXiv preprint arXiv:2204.01159},
      year={2022}
    }

## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed from the pytorch implementation of [CycleGAN-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

The code includes parts of the source code from [VNN](https://github.com/FlyingGiraffe/vnn) and [AtlasNet2](https://github.com/TheoDEPRELLE/AtlasNetV2) 
