# Mobile Image Colorization

This is a lightweight PyTorch implementation of paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511).

## Features



## Dependencies
- Python 3.6.8
- PyTorch 1.3

## Dataset

![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/imagenet.png)

Follow the [instruction](https://github.com/foamliu/ImageNet-Downloader) to download ImageNet dataset.




## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
Download [pre-trained model](https://github.com/foamliu/Colorful-Image-Colorization/releases/download/v1.0/model.06-2.5489.hdf5) into "models" folder then run:

```bash
$ python demo.py
```

Input | Output | GT | 
|---|---|---|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/0_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/0_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/0_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/1_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/1_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/2_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/2_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/3_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/3_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/3_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/4_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/4_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/4_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/5_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/5_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/6_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/6_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/7_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/7_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/7_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/8_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/8_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/8_gt.png)|
|![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/9_image.png) | ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/9_out.png)| ![image](https://github.com/foamliu/Colorful-Image-Colorization/raw/master/images/9_gt.png)|
