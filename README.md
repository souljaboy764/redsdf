# Regularized Deep Signed Distance Fields for Reactive Motion Generation
Puze Liu, Kuo Zhang, Davide Tateo, Snehal Jauhri, Jan Peters and Georgia Chalvatzaki
<p align="center">
<img src=https://git.ias.informatik.tu-darmstadt.de/ias_code/iros2022/redsdf/-/blob/main/fig/hri.gif width="400">
</p>

[website](https://irosalab.com/2022/02/28/redsdf/), paper

## Install
```python
pip install -e .
```

## Train Example Manifold (Static Model)
Change direction to the script path
```python
cd examples/static_manifold
```

There are two methods to generate dataset. First is generating dataset from mesh file: 
```python
python generate_data_from_mesh.py
```

Alternatively, the dataset is generated from pybullet.
```python
python generate_data_from_pybullet.py
```

Train the network. 
```python
python train_model.py
```

After training the network, the trained manifold can be shown as follows:
```python
python show_result.py --model_file path/to/the/trained/model
```

