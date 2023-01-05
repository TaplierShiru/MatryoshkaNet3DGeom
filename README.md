Re-implementation/improvement of Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers
===============================================================================

This re-implementation/improvement code from original [repo](https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/) from the paper  

**Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers**  
Stephan Richter and Stefan Roth. In CVPR 2018.  
[**Paper**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Richter_Matryoshka_Networks_Predicting_CVPR_2018_paper.pdf) [**Supplemental**](http://openaccess.thecvf.com/content_cvpr_2018/Supplemental/1524-supp.pdf)

Please cite original work if you use code or data from this/their repository.

Requirements and set up
-------------------------------------------------------------------------------
Clone the repository via `git clone https://github.com/TaplierShiru/MatryoshkaNet3DGeom`.

### Requirements
- python with version 3.7 and above
- numpy
- scipy
- pillow
- pytorch
- torchvision

General notes
-------------------------------------------------------------------------------
The shape layer representation will work the better the more consistent your
input shapes are wrt. occlusions and nesting of 3D shapes. Meshes from 
different sources will probably be not consistent and in this case fewer layers 
are likely to work better. Keep in mind that few layers can often reconstruct 
remarkably well. If mesh quality varies in the dataset (as in ShapeNet), you 
are probably better off using a single shape layer and increasing the number of 
inner residual blocks (`--block`) or number of inner feature channels (`--ngf`).

---

Datasets
-------------------------------------------------------------------------------
Original repository supports ShapeNet in 2 versions: as used in 3DR2N2[1], and as used
in PTN[2]. It also supports the highres car experiment from OGN[3]. To run it 
with the respective datasets, please check the [DatasetCollector.py](DatasetCollector.py). It commonly
expects only a base directory including sub directories for shapes and 
renderings. The renderings are expected to be 128x128 images (see below).

Adding a new dataset should be straightforward:

1. process images with [crop_images.py](crop_images.py).
2. convert binvox to voxel, voxel to shape layer with [voxel2layer](voxel2layer_torch.py).
3. write an adapter inheriting from [DatasetCollector](DatasetCollector.py), which collects samples

But if you try to run experiments from original repository - it won't work as expected due to errors/mistakes in original repo. Here I try to improve and fix this errors/mistakes. Below you can find detail information how to run certain dataset.

## 3DR2N2 Dataset
Download data from [here](https://github.com/chrischoy/3D-R2N2). We need [ShapeNet rendered images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) and [ShapeNet voxelized models](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz).

Download this two folders, unzip and put them into for example `3d_r2n2` folder.

Before training you need to prepare data for dataset. You need do next steps:

1. Convert `.binvox` files into `.mat` which are more suitable for training process. Command: 
```
python binvox2mat.py /path/to/3d_r2n2/ShapeNetVox32 -r -p -w=10
```

2. Convert `.mat` files into `.shl.mat` files which are encoded shape layers for training. Command: 
```
python voxel2layer.py /path/to/3d_r2n2/ShapeNetVox32 -r -p -w=10 -n=1
```

After that, we can start training process. My command to get suitable results:
```
python train.py --gpu=0 --basedir=/path/to/3d_r2n2/ --ncomp=1 --side=32 --batchsize=128 --down=3 --block=4 --ninf=8 --ngf=
512 --drop=20 --save_inter=1 --save_results=./results
```

Then training process is finished, we can explore results via `load_model_example_3dr2n2.ipynd`, where you can look at IoU plot and choose best model for test. Then you choose it, test command is:
```
python test.py --gpu 0 --set=test --file ./results/matryoshka_ShapeNet_666.pth.tar --basedir ./3d_r2n2 --ncomp=1 --save_results=./results
```

## Shapenet cars
Download data from [here](https://github.com/lmb-freiburg/ogn) ([direct link to dataset](http://lmb.informatik.uni-freiburg.de/data/ogn/data.zip)). Folder `ogn_octree/shapenet_cars` will contains only of `.ot` files aka voxels. Download scans/images from dataset above (only images) or [here](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz). To prepare dataset you need to do next steps:

1. Convert `.ot` to `.mat` files. Next command will convert all type of voxels (64^3, 128^3 and 256^3), but you can convert only one folder what you want (`/path/to/ogn_octree/shapenet_cars/256_l5` for example). Command:
```
python ot2mat.py /path/to/ogn_octree/shapenet_cars -r -p -w=10
```

2. Convert `.mat` files into `.shl.mat` files which are encoded shape layers for training. Command: 
```
python voxel2layer.py /path/to/ogn_octree/shapenet_cars -r -p -w=10 -n=5
```

My command to get suitable results:
```
python train.py --gpu=1 --shapenet_base_dir=/path/to/3d_r2n2/ShapeNetRendering --basedir=/path/to/ogn_octree/shapenet_cars --ncomp=5 --side=128 --batchsize=128 --down=4 --block=4 --ninf=8 --ngf=512 --drop=20 --save_inter=1 --val_inter=1 --epochs=50 --save_results=./results
_zoom --dataset=ShapeNetCars
```

## Faust dataset
Download data from [here](https://github.com/lmb-freiburg/ogn) ([direct link to dataset](http://lmb.informatik.uni-freiburg.de/data/ogn/data.zip)). Folder `faust` will contains only of `.ot` files aka voxels. Download scans/images from [here](http://faust.is.tue.mpg.de/). To preare dataset you need to do next steps:

1. You need merge this two folders, so `ogn_octree/faust` will contains of files from `MPI-FAUST` (there are `test` and `training` folders), just move these folders into `ogn_octree/faust`. At the end `ogn_octree/faust` folder will contains of folders: 
```
128_l4  256_l5  64_l4  test  training
```
2. Convert `.ot` to `.mat` files. Next command will convert all type of voxels (64^3, 128^3 and 256^3), but you can convert only one folder what you want (`/path/to/ogn_octree/faust/256_l5` for example). Command:
```
python ot2mat.py /path/to/ogn_octree/faust -r -p -w=10
```

3. Convert `.mat` files into `.shl.mat` files which are encoded shape layers for training. Command: 
```
python voxel2layer.py /path/to/ogn_octree/faust -r -p -w=10 -n=1
```

My command to get suitable results:
```
python train.py --gpu=0 --basedir=/path/to/ogn_octree/faust --ncomp=5 --side=128 --batchsize=16 --down=4 --block=4 --ninf=8
 --ngf=512 --drop=1000 --save_inter=100 --val_inter=100 --epochs=2000 --save_results=./results --dataset=Faust
```

---

Input images
-------------------------------------------------------------------------------
The networks are built to process input images of 128x128 pixels. 
For convenience, we provide a script that crops images to this size. 
Consequently, the *DatasetCollector* assumes that images are named `*.128.png` to 
indicate this format. Please have a look at [crop_images.py](crop_images.py) and 
[DatasetCollector](DatasetCollector.py).

### Different background colors and algorithm to cut image with window size
Tested on 3DR2N2 Dataset.

In the file `crop_images.py` you can find two methods:
```
def load_image(temp, alpha_map_to=255): 
    pass

def crop_image(img: PIL.Image, size: int, pad = 100, background_value=128, resample=PIL.Image.LANCZOS):
    pass
```

With different `alpha_map_to` and `background_value` values - different final image result:

`alpha_map_to` - color of the background;

`background_value` - values of the padded values for image;

In the original repo background value and alpha map equal to 255 both (so image has white background and zoomed). But if you trained images not processed with crop algorithm in the original repo (so images has alpha channel) or image has alpha channel when if image loaded alpha channel will be equal to grey (128) value but crop will be with white (255) value. As a result final image will be with background 128 and NOT zoomed. 

In first iteration I trained model with grey and NOT zoomed config. So my results were not good enough compare to original paper.

Here are examples with result of the model training on 3DR2N2 Dataset measured with IoU with different configurations:

![crop-example](/images/matreshka.png)

From these experiments final output: using white background the final training has better metric score, and zoomed image has worse results. But the difference is small, so I think its does not matter.

Original repo use configuration with back color equal to 255 and zoomed. This repo have same configuration.

References
-------------------------------------------------------------------------------
[1] C. B. Choy, D. Xu, J. Gwak, K. Chen, and S. Savarese. 
    3D-R2N2: A unified approach for single and multi-view 3D object 
    reconstruction. ECCV 2016
	
[2] X. Yan, J. Yang, E. Yumer, Y. Guo, and H. Lee. 
    Perspective transformer nets: Learning single-view 3D object reconstruction
    without 3D supervision. NIPS 2016
	
[3] M. Tatarchenko, A. Dosovitskiy, and T. Brox. 
    Octree generating networks: Efficient convolutional architectures for
    high-resolution 3D outputs. ICCV 2017
