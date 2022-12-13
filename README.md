This repository (re)implements the code for the paper [**Overparameterized ReLU Neural Networks Learn the Simplest Model: Neural Isometry and Phase Transitions**](https://arxiv.org/abs/2209.15265) (Wang et al 2022).

# Introduction

Suppose that $\mathbf{X}\in \mathbb{R}^{n\times d}$ and $\mathbf{y}\in \mathbb{R}^d$ are the data matrix and the label vector respectively. In the paper, we focus on the following two-layer neural networks with $m$ hidden neurons:

## ReLU networks

  
$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) = (\mathbf{X}{\mathbf{W}}^{(1)})_+ \mathbf{w}^{(2)},
$$

where $\Theta = (\mathbf{W}^{(1)},\mathbf{w}^{(2)})$, $\mathbf{W}^{(1)} \in \mathbb{R}^{d\times m}$ and $\mathbf{w}^{(2)} \in \mathbb{R}^{m}$.

## ReLU networks with skip connections

  
$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) =\mathbf{X}\mathbf{w}^{(1)}_{1} w^{(2)}_{1}+\sum_{i=2}^m (\mathbf{X}\mathbf{w}^{(1)}_{i})_+ w^{(2)}_{i},
$$

where $\Theta = (\mathbf{W}^{(1)},\mathbf{w}^{(2)})$, $\mathbf{W}^{(1)}\in \mathbb{R}^{d\times m}$ and $\mathbf{w}^{(2)}\in \mathbb{R}^{m}$.

## ReLU networks with batch normalization

$$
f^\mathrm{ReLU}(\mathbf{X};\Theta) =\sum_{i=1}^m \operatorname{NM}_{\alpha_i}((\mathbf{X}\mathbf{w}^{(1)}_{i})_+)w^{(2)}_{i},
$$

where $\Theta = (\mathbf{W}^{(1)},\mathbf{w}^{(2)},\mathbf{\alpha})$ and the normalization operation $\operatorname{NM}_\alpha(\mathbf{v})$ is defined by

$$
\operatorname{NM}_{\alpha}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}\alpha, \quad \mathbf{v}\in\mathbb{R}^n,\alpha\in \mathbb{R}.
$$

## Training setup

We consider the regularized training problem 

$$
\min_{\Theta} \frac{1}{2}\|f(\mathbf{X};\Theta)-\mathbf{y}\|_2^2+\frac{\beta}{2}R(\Theta).
$$

When $\beta\to 0$, the optimal solution of the above problem solves the following minimal norm problem

$$
    \min_{\Theta} R(\Theta), \text{ s.t. } f(\mathbf{X};\Theta)=\mathbf{y}.
$$

We include code to solve convex optimization formulations of the minimal norm problem and to train neural networks discussed in the paper, respectively. We also include code to plot the phase transition graphs shown in the paper. 

More details about the numerical experiments can be found in the appendix of the paper.

# Requirements

When solving convex programs, [CVXPY](https://www.cvxpy.org/install/index.html) (version>=1.1.13) is needed. [Mosek](https://www.mosek.com/downloads/) solver is preferred. To use Mosek, you will need to register for an account and download a license. You can also change the solver according to the documentation of CVXPY.

When training neural networks discussed in the paper, [PyTorch](https://pytorch.org/get-started/locally/) (version>=1.10.0) is needed.

# Usage

Compute the recovery rate of the planted linear neuron by solving the minimal norm problem for ReLU networks with skip connections over 5 independent trials. 
```bash
python rec_rate_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0
```

Compute the absolute distance by solving the convex programs over 5 independent trials. 
```bash
# minimal norm problem
python minnrm_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0

# convex training problem
python cvx_train_skip.py --n 400 --d 100 --sample 5 --sigma 0 --optw 0 --optx 0
```

Compute the test distance by training ReLU networks with skip connections over 10 independent trials. 
```bash
python ncvx_train_skip.py --n 400 --d 100 --sample 10 --sigma 0 --optw 0 --optx 0
```

- You can change `--save_details`, `--save_folder`, `--seed` accordingly. 
- For ReLU networks with normalization layer, you can also set the number of planted neurons by changing `--neu`. Details about the supported types of planted neuron and data matrix can be found in the comments of the code.

# Codebase

The codebase is structured as follows:

- `main.py`: the main CLI.
- `nic.py`: Neural Isometry Condition implementations.
- `plot.py`: for plotting data, as well as a CLI that plots results from a NumPy file containing a numpy array of shape (n, d, sample).
- `training/`: training utilities for solving the actual optimization problems.
    - `common.py`: helpers for formulating the convex optimization problems.
    - `convex_program.py`: the abstract superclass for implementing the convex problems.
    - `ncvx_network_train.py`: nonconvex neural network training code with PyTorch.
    - `networks.py`: simple neural networks implemented in PyTorch.
    - `normalized.py`: the implementation for the convex formulation of a ReLU network with batch normalization.
    - `skip.py`: the implementation for the convex formulation of a ReLU network with a skip connection. Can also be used for a plain network with no skip connection.

# Maintainers

Originally implemented by:

- Yixuan Hua (yh7422@princeton.edu)
- Yifei Wang (wangyf18@stanford.edu)

This fork is a reimplementation by

- Alexander Cai (alexcai@college.harvard.edu)
- Max Nadeau (mnadeau@college.harvard.edu)

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

# License

[MIT](https://choosealicense.com/licenses/mit/)

