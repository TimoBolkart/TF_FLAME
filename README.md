## FLAME: Faces Learned with an Articulated Model and Expressions (TF)

[FLAME](http://flame.is.tue.mpg.de/) is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. This repository provides sample Tensorflow code to experiment with the FLAME model. Parts of the repository are adapted from the [Chumpy](https://github.com/mattloper/chumpy)-based [FLAME-fitting repository](https://github.com/Rubikplayer/flame-fitting). 

<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME combines a linear identity shape space (trained from 3800 scans of human heads) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please about the model, please see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf) and the [supplementary video](https://youtu.be/36rPTkhiJTM).

### Content

This repository demonstrates how to 
1) sample 3D face meshes 
2) fit the 3D model to 2D landmarks
3) fit the 3D model to 3D landmarks 
4) fit the 3D model to registered 3D meshes
5) how to generate templates for speech-driven facial animation ([VOCA](https://github.com/TimoBolkart/voca))

### Set-up

The has been tested with Python3.6, using Tensorflow 1.15.2.

Install pip and virtualenv

```
sudo apt-get install python3-pip python3-venv
```

Clone the git project:
```
$ git clone https://github.com/TimoBolkart/TF_FLAME.git
```

Set up virtual environment:
```
$ mkdir <your_home_dir>/.virtualenvs
$ python3 -m venv <your_home_dir>/.virtualenvs/flame
```

Activate virtual environment:
```
$ cd TF_FLAME
$ source <your_home_dir>/flame/bin/activate
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

Download the FLAME model from [MPI-IS/FLAME](http://flame.is.tue.mpg.de/). You need to sign up and agree to the model license for access to the model and the data.<br/>


### Demo

We provide demos to i) draw random samples from FLAME to demonstrate how to edit the different FLAME parameters, ii) to fit FLAME to 3D landmarks, iii) to fit FLAME to a registered 3D mesh (i.e. in FLAME topology), and iv) to generate [VOCA](https://github.com/TimoBolkart/voca) templates.


##### Sample FLAME

This demo introduces the different FLAME parameters (i.e. pose, shape, expression, and global transformation) of the FLAME model by generating random sample meshes. Please note that this does not demonstrate how to get realistic 3D face samples from the model.
```
python sample_FLAME.py
```

##### Fit 2D landmarks

This demo demonstrates how to fit FLAME to 2D landmarks. Corresponding 2D landmarks can for instance be automatically predicted using [2D-FAN Torch](https://github.com/1adrianb/2D-and-3D-face-alignment) or [2D-FAN Pytorch](https://github.com/1adrianb/face-alignment). (The test images are taken from CelebA-HQ) 
```
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --template_fname './data/template.ply' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data.npy' --target_img_path './data/imgHQ00088.jpeg' --target_lmk_path './data/imgHQ00088_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --template_fname './data/template.ply' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data.npy' --target_img_path './data/imgHQ00095.jpeg' --target_lmk_path './data/imgHQ00095_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/male_model.pkl' --template_fname './data/template.ply' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data.npy' --target_img_path './data/imgHQ00039.jpeg' --target_lmk_path './data/imgHQ00039_lmks.npy' --out_path './results'
python fit_2D_landmarks.py --model_fname './models/female_model.pkl' --template_fname './data/template.ply' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data.npy' --target_img_path './data/imgHQ01148.jpeg' --target_lmk_path './data/imgHQ01148_lmks.npy' --out_path './results'

```

##### Create textured mesh

This demo demonstrates how to create a textured mesh in FLAME topology by projecting an image onto the fitted FLAME mesh (i.e. obtained by fitting FLAME to 2D landmarks). (The test images are taken from CelebA-HQ)
```
python build_texture_from_image.py --source_img './data/imgHQ00088.jpeg' --target_mesh './results/imgHQ00088.obj' --target_scale './results/imgHQ00088_scale.npy' --texture_mapping './data/texture_data.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ00095.jpeg' --target_mesh './results/imgHQ00095.obj' --target_scale './results/imgHQ00095_scale.npy' --texture_mapping './data/texture_data.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ00039.jpeg' --target_mesh './results/imgHQ00039.obj' --target_scale './results/imgHQ00039_scale.npy' --texture_mapping './data/texture_data.npy' --out_path './results'
python build_texture_from_image.py --source_img './data/imgHQ01148.jpeg' --target_mesh './results/imgHQ01148.obj' --target_scale './results/imgHQ01148_scale.npy' --texture_mapping './data/texture_data.npy' --out_path './results'
```

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


##### Generate VOCA template

[VOCA](https://github.com/TimoBolkart/voca) is a framework to animate a static face mesh in FLAME topology from speech. This demo samples the FLAME identity space to generate new templates that can then be animated with VOCA. 
```
python sample_FLAME.py --option sample_VOCA_template
```


### Supported projects

FLAME supports several projects such as
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image without 3D Supervision](https://github.com/soubhiksanyal/RingNet)
* [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://github.com/vchoutas/smplify-x)

FLAME is part of SMPL-X
* [SMPL-X: A new joint 3D model of the human body, face and hands together](https://github.com/vchoutas/smplx)


## License

FLAME is available under [Creative Commons Attribution license](https://creativecommons.org/licenses/by/4.0/). By using the model or the code code, you acknowledge that you have read the license terms (http://flame.is.tue.mpg.de/model_license), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code.


### Citing

When using this code in a scientific publication, please cite 
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

## Acknowledgement

The Tensorflow implementation used in this project is adapted from [HMR](https://github.com/akanazawa/hmr). We thank Angjoo Kanazawa for making this code available.
We thank Ahmed Osman for support with Tensorflow.

