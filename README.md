# FACT-AI 2019-2020

## Reproduction of Paper: Towards Robust Interpretability with Self-Explaining Neural Networks
This repo contains models, their performance, visualizations and explainations. New models can not be trained with this repository. An extension of the visualization of basic concepts is added.

## Repo structure
- Root: presentation, notebook, report, and images generated from code

- SENN: code, which includes all utils used in the notebook 

> Note: many functionalities were taken from https://github.com/dmelis/SENN
## How to work with this repo
1. Make sure you have python 3.6 installed (if not, we cannot guarantee that the code will work)
```
$ python --version
Python 3.6.10
```

2. We recommend setting up a virtual environment for python 3.6
```
$ python -m venv .venv
$ source .venv/bin/activate
```
3. Clone the repo:
```
$ git clone git@github.com:ricokoff/FACT-AI.git
```

4. Enter the SENN folder and use:
```
$ pip install -r requirements.txt
$ python setup.py install
```

5. Make sure you have Jupyter installed and take a look at our notebook.
If you don't we highly recommend to install Jupyter while inside your environment.
```
$ pip install ipython jupyter ipykernel
$ python -m ipykernel install --user --name SENN
```

Once you run Jupyter Notebooks, you can now see your virtual environment available as a kernel:
![SENN Kernel](https://i.imgur.com/4m9Atnp.png)


> Note: this work has been tested only with the use of a python 3.6 environment with Jupyter and NOT with the use of Anaconda/Miniconda/etc., so using this might not work
      

## Contributors:
- Ewoud Vermeij - 11348860 - ewoudvermeij@gmail.com

- Yke Rusticus - 11306386 - yke.rusticus@student.uva.nl

- Rico Mossinkoff - 12805157 - ricokoff@hotmail.com

- Roberto Schiavone - 12883980 - r.schiavone@student.vu.nl

## TA:
- Simon Passenheim

