# Regularized Deep Signed Distance Fields for Reactive Motion Generation
Puze Liu, Kuo Zhang, Davide Tateo, Snehal Jauhri, Jan Peters and Georgia Chalvatzaki
<p align="center">
<img src=fig/hri.gif width="400">
</p>

[website](https://irosalab.com/2022/02/28/redsdf/), [paper](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/IROS_2022_ReDSDF.pdf)

## Installation
```python
pip install -e .
```

## Data Augmentation
We provide two ways to generate the dataset. We can generate the dataset directly from the mesh file
when the mesh is dense. Alternatively we can use **pybullet** simulator to generate augmented dataset. 
#### Dense Mesh File --- Direct Generation
For the mesh file dense point clouds, such as "object_models/human.obj", 
we can obtain the normals of the point cloud directly from the mesh. For example:  
```python
python examples/static_manifold/generate_data_from_pybullet.py --mesh_file object_models/human.obj --save_dir [SAVE_DIR] 
```

#### Sparse Mesh or Articulated Objects --- Simulator Generation
For Sparse Mesh or articulated multi-body objects. 
The normals of the surfaces are hard to obtain from directly from mesh, we can use simulator to obtain the point cloud 
and the normals of the objects surface.

For **sparse** mesh, such as object_models/sofa.obj:
```python
python examples/static_manifold/generate_data_from_pybullet.py --mesh_file object_models/sofa.obj --save_dir [SAVE_DIR]
```

For **Articulated Objects**, such as **Tiago Robot**, we uniformly sample the 10000 feasible configurations (This generation 
takes approx. 2.5 hours on AMD® Ryzen 9 3900x):
```python
python examples/tiago_manifold/generate_dataset.py --urdf_dir object_models/tiago_urdf --save_dir [SAVE_DIR] --n_poses 10000 
```

For **Human SMPL Model**, please visit [SMPL-X](https://smpl-x.is.tue.mpg.de/) and [AMASS](https://amass.is.tue.mpg.de/) to 
obtain the mesh model and the human motion dataset.

## Training

```python
python examples/[MANIFOLD_TYPE]/train_model.py
```
Note that the difference here is the choice of the object **center** and 
**batch_size** depending on the size of the dataset

After training the network, the trained manifold can be visualized by with following command:
```python
python examples/[MANIFOLD_TYPE]/plot_model.py --model_file [MODEL_FILE]
```

## Reactive Control
For reactive control, we provide the pretrained_model in "trained_sdf"


Run the experiment of whole body control:
```python
python reactive_control/whole_body_control.py --debug_gui --use_cuda
```

Alternatively, run the experiment of human robot interaction:
```python
python reactive_control/shared_workspace.py --debug_gui --use_cuda
```
