# Face2to3D
3D Facial Reconstruction
资源及相关链接放在[material](https://github.com/changebio/Face2to3D/tree/master/material)文件夹

代码放在[sources](https://github.com/changebio/Face2to3D/tree/master/sources)文件夹

自己代码可以在[try](https://github.com/changebio/Face2to3D/tree/master/try)文件夹中以自己的名字命名一个文件夹

## 重要信息
jupyter notebook address

http://106.75.34.228:82/29601f2d-b8df-47ec-9b10-a006b4e0cef3/?token=a2f7514a1f014c556b0d758db5a738747d761e4702b0cd6e
```
#Jupyter workflow in remote server after login jupyter
data -> Face2to3D -> try ->your_code_directory
#Terminal workflow in remote server
New -> Terminal
cd /data/data/Face2to3D/
```
#### 注意事项
1.尽量不要把**代码或文件**放到data文件夹外。因为可能**丢失文件**。

2.在**try**文件夹中的子项目文件夹写和修改代码。尽量不要在sources中修改代码。

3.合作项目代码和资源尽量更新到GitHub，避免大家做重复工作。

**4.记着自己打开的Jupyter Notebook和terminal，避免误会其他人的。（Jupyter Notebook命名规则：名字简写_其他，如 hy_3ddfa.ipynb**


## State-of-Art Models Code
[3DDFA](https://github.com/changebio/Face2to3D/tree/master/sources/3DDFA)

[PRNet](https://github.com/changebio/Face2to3D/tree/master/sources/PRNet) [pytorch version](https://github.com/changebio/Face2to3D/tree/master/sources/pytorch-prnet) [TF training code](https://github.com/changebio/Face2to3D/tree/master/sources/training_codes_for_PRNet_3D_Face)

[Nonlinear 3DMM](https://github.com/changebio/Face2to3D/tree/master/sources/Nonlinear_Face_3DMM)

## 项目说明
吾日三省吾身：白否？富否？美否？ 我们将利用三维人脸重建技术，更好地“审视”自己。三维人脸重建是3D计算机视觉中的重要部分，可以广泛应用于三维人脸识别、辅助医疗、个性化3D打印、影视特效、虚拟现实等领域。传统三维人脸重建技术需要根据多视图估计相机运动，并计算每张图像的深度图，需要消耗大量时间及计算资源。利用深度摄像头可以更快速获取图像的深度，在一定程度上减少重建时间，然而对于普通摄像头而言，如何快速重建出三维人脸模型？近年来，随着深度学习技术的发展，从视频中快速重建出三维人脸模型已经变得可能。看腻了二维人脸图像，不妨来挑战一下更高级的3D人脸建模。 
本课题涉及：人脸检测；人脸特征点定位；亚洲人脸模型构建；人脸重建。

![项目导图](https://github.com/changebio/Face2to3D/blob/master/material/IMG_2956.JPG)

Latest Network Architecture and Performance Comparison
![Performance](https://user-images.githubusercontent.com/8948023/56006880-36622000-5d09-11e9-9465-8d52e3433d5f.png)

## 数据

**3D Morphable Model (3DMM)**

- Basel Face Model (BFM)
	- BFM2009 (53490vertices) 
	- BFM2017 (53149vertices)

- Surrey Face Model (SFM)
	- SFM1724
	- SFM3448
	- SFM16759 (No texture) 
	- SFM29587 (No texture)

- Large scale Face Model (LSFM) 

## 计划

## 小组

## 建议

## Tools
### common git command
```
git status
git add .
git commit -m "xxxxxx"

#pull and push code from GitHub
git pull #equal to git fetch; git merge
git push
```

### basic markdown format

https://help.github.com/en/articles/basic-writing-and-formatting-syntax
