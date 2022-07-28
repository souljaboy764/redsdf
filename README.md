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

### Train Example Manifold (Static Model)
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

Train the network:
```python
python train_model.py
```

After training the network, the trained manifold can be shown with following command:
```python
python show_result.py --model_file path/to/the/trained/model
```

### Train Example Manifold (Dynamic Model)
There are two dynamic models that can be trained: human body and robot TIAGo++. If you want to train a human model, please ask the author for the SMPL dataset. Here we give the example of training model of TIAGo++. It takes a very long time.
```python
cd examples/tiago_manifold
```

Generate dataset:
```python
python generate_dataset.py
```

Train the network:
```python
python train_model.py --use_cuda
```

Show results:
```python
python plot_model.py --model_file path/to/the/trained/model
```

### Run Experiment
Change direction to the script path
```python
cd reactive_control
```

Run the experiment of whole body control:
```python
python whole_body_control.py --debug_gui --use_cuda
```

Alternatively, run the experiment of human robot interaction:
```python
python shared_workspace.py --debug_gui --use_cuda
```

