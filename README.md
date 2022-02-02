# FLAME: Articulated Expressive 3D Head Model (TF)

This is an official Tensorflow-based [FLAME](http://flame.is.tue.mpg.de/) repository. 

We also provide [PyTorch FLAME](https://github.com/HavenFeng/photometric_optimization), a [Chumpy](https://github.com/mattloper/chumpy)-based [FLAME-fitting repository](https://github.com/Rubikplayer/flame-fitting), and code to [convert from Basel Face Model to FLAME](https://github.com/TimoBolkart/BFM_to_FLAME).

<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. FLAME combines a linear identity shape space (trained from head scans of 3800 subjects) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)

```
Learning a model of facial shape and expression from 4D scans
Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero
ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 2017
```
and the [supplementary video](https://youtu.be/36rPTkhiJTM).

### Content

This repository demonstrates how to 
1) sample 3D face meshes 
2) fit the 3D model to 2D landmarks
3) fit the 3D model to 3D landmarks 
4) fit the 3D model to registered 3D meshes
5) sample the texture space
6) how to generate templates for speech-driven facial animation ([VOCA](https://github.com/TimoBolkart/voca))

### Set-up

The code has been tested with Python3.6, using Tensorflow 1.15.2.

Install pip and virtualenv

```
sudo apt-get install python3-pip python3-venv
```

Clone the git project:
```
git clone https://github.com/TimoBolkart/TF_FLAME.git
```

Set up virtual environment:
```
mkdir <your_home_dir>/.virtualenvs
python3 -m venv <your_home_dir>/.virtualenvs/TF_FLAME
```

Activate virtual environment:
```
cd TF_FLAME
source <your_home_dir>/.virtualenvs/TF_FLAME/bin/activate
```

Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh) within the virtual environment.

Make sure your pip version is up-to-date:
```
pip install -U pip
```

Other requirements (including tensorflow) can be installed using:
```
pip install -r requirements.txt
```

The visualization uses OpenGL which can be installed using:
```
sudo apt-get install python-opengl
```

### Data

Download the FLAME model and the MPI texture space from [MPI-IS/FLAME](https://flame.is.tue.mpg.de/downloads). You need to sign up and agree to the model license for access to the model and the data. Further, download the [FLAME_texture_data](http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip) and unpack this into the data folder. If you want to use a statistical appearance texture space for FLAME, download either [AlbedoMM (CVPR 2020)](https://github.com/waps101/AlbedoMM) or the [FLAME texture space](https://flame.is.tue.mpg.de).<br/>

### Demo

We provide demos to 
i) draw random samples from FLAME to demonstrate how to edit the different FLAME parameters, 
ii) to fit FLAME to 3D landmarks, 
iii) to fit FLAME to a registered 3D mesh (i.e. in FLAME topology), and 
iv) to generate [VOCA](https://github.com/TimoBolkart/voca) templates.

##### Sample FLAME

This demo introduces the different FLAME parameters (i.e. pose, shape, expression, and global transformation) of the FLAME model by generating random sample meshes. Please note that this does not demonstrate how to get realistic 3D face samples from the model.
```
python sample_FLAME.py --option sample_FLAME --model_fname './models/generic_model.pkl' --num_samples 5 --out_path './FLAME_samples'
```
By default, running this demo uses an OpenGL-based mesh viewer viewer to visualize the samples. If this causes any problems, try running the demo with the additional flag --visualize False to disable the visualization.

##### Fit 2D landmarks

This demo demonstrates how to fit FLAME to 2D landmarks. Corresponding 2D landmarks can for instance be automatically predicted using [2D-FAN Torch](https://github.com/1adrianb/2D-and-3D-face-alignment) or [2D-FAN Pytorch](https://github.com/1adrianb/face-alignment). (The test images are taken from CelebA-HQ) 
```
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data_512.npy' --target_img_path './data/imgHQ00088.jpeg' --target_lmk_path './data/imgHQ00088_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data_512.npy' --target_img_path './data/imgHQ00095.jpeg' --target_lmk_path './data/imgHQ00095_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/male_model.pkl' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data_512.npy' --target_img_path './data/imgHQ00039.jpeg' --target_lmk_path './data/imgHQ00039_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data_512.npy' --target_img_path './data/imgHQ01148.jpeg' --target_lmk_path './data/imgHQ01148_lmks.npy' --out_path './results'
```
By default, running the demo opens a window to visualize the fitting progress. This will fail if running the code remotely. In this case try running the demo with an additional flag ```--visualize False``` to disable the visualization. If you want to get FLAME textures of resolutions other than 512x512, use texture_data_256.npy, texture_data_1024.npy, or texture_data_2048.npy instead. 

##### Create textured mesh

This demo demonstrates how to create a textured mesh in FLAME topology by projecting an image onto the fitted FLAME mesh (i.e. obtained by fitting FLAME to 2D landmarks). (The test images are taken from CelebA-HQ)
```
python build_texture_from_image.py --source_img './data/imgHQ00088.jpeg' --target_mesh './results/imgHQ00088.obj' --target_scale './results/imgHQ00088_scale.npy' --texture_mapping './data/texture_data_512.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ00095.jpeg' --target_mesh './results/imgHQ00095.obj' --target_scale './results/imgHQ00095_scale.npy' --texture_mapping './data/texture_data_512.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ00039.jpeg' --target_mesh './results/imgHQ00039.obj' --target_scale './results/imgHQ00039_scale.npy' --texture_mapping './data/texture_data_512.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ01148.jpeg' --target_mesh './results/imgHQ01148.obj' --target_scale './results/imgHQ01148_scale.npy' --texture_mapping './data/texture_data_512.npy' --out_path './results'
```
If you want to get FLAME textures of resolutions other than 512x512, use texture_data_256.npy, texture_data_1024.npy, or texture_data_2048.npy instead. 

##### Fit 3D landmarks

This demo demonstrates how to fit FLAME to 3D landmarks. Corresponding 3D landmarks can for instance be selected manually from 3D scans using [MeshLab](http://www.meshlab.net/). 
```
python fit_3D_landmarks.py
```

##### Fit registered 3D meshes

This demo shows how to fit FLAME to a 3D mesh in FLAME topology (i.e. in dense corresponcence to the model template). Datasets with available meshes in FLAME topology are e.g. [registered D3DFACS](http://flame.is.tue.mpg.de/), [CoMA dataset](http://coma.is.tue.mpg.de/), and [VOCASET](http://voca.is.tue.mpg.de/).
```
python fit_3D_mesh.py
```
Note that this demo to date does not support registering arbitrary 3D face scans. This requires replacing the vertex loss function by some differentiable scan-to-mesh or mesh-to-scan distance.

##### Sample texture space

Three texture spaces are available for FLAME, the [MPI texture space](https://flame.is.tue.mpg.de/downloads), [AlbedoMM](https://github.com/waps101/AlbedoMM), and the [BFM color space](https://github.com/TimoBolkart/BFM_to_FLAME). This demo generates FLAME meshes with textures randomly sampled from the MPI texture space (download [here](https://flame.is.tue.mpg.de/downloads))
```
python sample_texture.py --model_fname './models/generic_model.pkl' --texture_fname './models/FLAME_texture.npz' --num_samples 5 --out_path './texture_samples_MPI'
```

Randomly sample textures from the [AlbedoMM](http://openaccess.thecvf.com/content_CVPR_2020/papers/Smith_A_Morphable_Face_Albedo_Model_CVPR_2020_paper.pdf) texture space (download  albedoModel2020_FLAME_albedoPart.npz [here](https://github.com/waps101/AlbedoMM/releases))
```
python sample_texture.py --model_fname './models/generic_model.pkl' --texture_fname './models/albedoModel2020_FLAME_albedoPart.npz' --num_samples 5 --out_path './texture_samples_AlbedoMM'
```

##### Generate VOCA template

[VOCA](https://github.com/TimoBolkart/voca) is a framework to animate a static face mesh in FLAME topology from speech. This demo samples the FLAME identity space to generate new templates that can then be animated with VOCA. 
```
python sample_FLAME.py --option sample_VOCA_template --model_fname './models/generic_model.pkl' --num_samples 5 --out_path './FLAME_samples'
```
By default, running this demo uses an OpenGL-based mesh viewer viewer to visualize the samples. If this causes any problems, try running the demo with the additional flag --visualize False to disable the visualization.

### Landmarks

<p align="center"> 
<img src="data/landmarks_51_annotated.png" width="50%">
</p>

The provided demos fit FLAME to 3D landmarks or to a scan, using 3D landmarks for initialization and during fitting. Both demos use the shown 51 landmarks. Providing the landmarks in the exact order is essential. The landmarks can for instance be obtained with [MeshLab](https://www.meshlab.net/) using the PickPoints module. PickPoints outputs a .pp file containing the selected points. The .pp file can be loaded with the provided 'load_picked_points(fname)' function in utils/landmarks.py.

### Citing

When using this code in a scientific publication, please cite FLAME 
```
@article{FLAME:SiggraphAsia2017,
  title = {Learning a model of facial shape and expression from {4D} scans},
  author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  year = {2017},
  url = {https://doi.org/10.1145/3130800.3130813}
}
```

### License

The FLAME model is under a Creative Commons Attribution license. By using this code, you acknowledge that you have read the terms and conditions (https://flame.is.tue.mpg.de/modellicense.html), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code. You further agree to cite the FLAME paper when reporting results with this model.

### Supported projects

FLAME supports several projects such as
* [CoMA: Convolutional Mesh Autoencoders](https://github.com/anuragranj/coma)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image without 3D Supervision](https://github.com/soubhiksanyal/RingNet)
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://github.com/vchoutas/smplify-x)
* [ExPose: Monocular Expressive Body Regression through Body-Driven Attention](https://github.com/vchoutas/expose)
* [GIF: Generative Interpretable Faces](https://github.com/ParthaEth/GIF)
* [DECA: Detailed Expression Capture and Animation](https://github.com/YadiraF/DECA)

FLAME is part of [SMPL-X: : A new joint 3D model of the human body, face and hands together](https://github.com/vchoutas/smplx)

### Acknowledgement

The Tensorflow implementation used in this project is adapted from [HMR](https://github.com/akanazawa/hmr). We thank Angjoo Kanazawa for making this code available.
We thank Ahmed Osman for support with Tensorflow.

