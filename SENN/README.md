# SENN
Self-Explaining Neural Networks

## Installation
It's highly recommended that the following steps be done **inside a virtual 
environment** (e.g., via `virtualenv` or `anaconda`).
```
python -m venv .
```

#### Install prereqs
```
pip install -r requirements.txt
python setup.py install
```
## How to use
To train models from scratch:
```
python scripts/main_mnist.py --train
```

To use pretrained models:
```
python scripts/main_mnist.py
```

## Overall Code Structure
* aggregators.py - defines the Aggregation functions
* conceptizers.py - defines the functions that encode inputs into concepts (h(x))
* parametrizers.oy - defines the functions that generate parameters from inputs (theta(x))
* trainers.py - objectives, losses and training utilities
