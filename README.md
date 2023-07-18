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
f^\mathrm{ReLU-skip}(\mathbf{X};\Theta) =\mathbf{X}\mathbf{w}^{(1)}_{1} w^{(2)}_{1}+\sum_{i=2}^m (\mathbf{X}\mathbf{w}^{(1)}_{i})_+ w^{(2)}_{i},
$$

where $\Theta = (\mathbf{W}^{(1)},\mathbf{w}^{(2)})$, $\mathbf{W}^{(1)}\in \mathbb{R}^{d\times m}$ and $\mathbf{w}^{(2)}\in \mathbb{R}^{m}$.

## ReLU networks with batch normalization

$$
f^\mathrm{ReLU-norm}(\mathbf{X};\Theta) =\sum_{i=1}^m \operatorname{NM}_{\alpha_i}((\mathbf{X}\mathbf{w}^{(1)}_{i})_+)w^{(2)}_{i},
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

# Usage and reproducing figures

## Equations

| Learned model | Equation  | Formulation                   |
| ------------- | --------- | ----------------------------- |
| ReLU          | exact     | 11 (top of p. 8)              |
| ReLU          | approx    | 211 (skip connection removed) |
| ReLU-skip     | exact     | 6 (top of p. 5)*              |
| ReLU-skip     | nonconvex | 9 (bottom of p. 7)            |
| ReLU-skip     | relaxed   | 15 (bottom of p. 8)           |
| ReLU-skip     | approx    | 211 (bottom of p. 57)         |
| ReLU-norm     | exact     | 16 (top of page 9)            |
| ReLU-norm     | relaxed   | 17 (middle of page 9)         |
| ReLU-norm     | approx    | 212 (bottom of page 9)        |

*We implement this with the w_0 norm added to the objective function. We suspect this was a typo in the paper.

## Training figures

| Figure  | Planted model | Equation | Metric            | Command (after `python main.py`)                                           | Image |
| ------- | ------------- | -------- | ----------------- | -------------------------------------------------------------------------- | ----- |
| 2       | linear        | 6        | test distance     | `linear skip exact`                                                        | ![](./results/learned_skip/planted_linear/form_exact/trial__n100__d50__w1__k1__X0__stdev0.0__sample5/test_err.png) |
| 4a      | linear        | 15       | recovery          | `linear skip relaxed`                                                      | ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w1__k1__X0__stdev0.0__sample5/recovery.png) |
| 4b      | linear        | 6        | recovery          | `linear skip exact`                                                        | ![](./results/learned_skip/planted_linear/form_exact/trial__n100__d50__w1__k1__X0__stdev0.0__sample5/recovery.png) |
| 5a,5b   | ReLU-norm     | 17       | recovery          | `--k {2,3} normalized normalized relaxed`                                  | ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w2__k2__X0__stdev0.0__sample5/recovery.png) ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w2__k3__X0__stdev0.0__sample5/recovery.png) |
| 6       | linear        | 15       | recovery          | `--optx {0,1,2,3} --optw 0 linear skip relaxed`                            | ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w0__k1__X0__stdev0.0__sample5/recovery.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w0__k1__X1__stdev0.0__sample5/recovery.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w0__k1__X2__stdev0.0__sample5/recovery.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w0__k1__X3__stdev0.0__sample5/recovery.png) |
| 7       | linear        | 15       | weight distance   | `--sigma {0,0.05,0.1,0.2} linear skip relaxed`                             | ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w1__k1__X0__stdev0.0__sample5/dis_abs.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w1__k1__X0__stdev0.05__sample5/dis_abs.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w1__k1__X0__stdev0.1__sample5/dis_abs.png) ![](./results/learned_skip/planted_linear/form_relaxed/trial__n100__d50__w1__k1__X0__stdev0.2__sample5/dis_abs.png) |
| 8       | linear        | 9        | test error        | `--sigma {0,0.05,0.1,0.2} linear skip gd`                                  | ![](./results/learned_skip/planted_linear/form_gd/trial__n100__d50__w1__k1__X0__stdev0.0__sample5/test_err.png) ![](./results/learned_skip/planted_linear/form_gd/trial__n100__d50__w1__k1__X0__stdev0.05__sample5/test_err.png) ![](./results/learned_skip/planted_linear/form_gd/trial__n100__d50__w1__k1__X0__stdev0.1__sample5/test_err.png) ![](./results/learned_skip/planted_linear/form_gd/trial__n100__d50__w1__k1__X0__stdev0.2__sample5/test_err.png) |
| 9       | ReLU-norm     | 17       | weight distance   | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 3 normalized normalized relaxed`    | ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w3__k2__X0__stdev0.0__sample5/dis_abs.png) ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w3__k2__X0__stdev0.05__sample5/dis_abs.png) ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w3__k2__X0__stdev0.1__sample5/dis_abs.png) ![](./results/learned_normalized/planted_normalized/form_relaxed/trial__n100__d50__w3__k2__X0__stdev0.2__sample5/dis_abs.png) |
| 18      | linear        |          | recovery          | `--optx {0,1,2,3} linear skip relaxed`                                     |
| 19      | linear        |          | test distance     | `--sigma {0,0.05,0.1,0.2} linear skip relaxed`                             |
| 20      | linear        |          | weight distance   | `--sigma {0,0.05,0.1,0.2} linear skip relaxed`                             |
| 21      | linear        |          | test distance     | `--sigma {0,0.05,0.1,0.2} linear skip relaxed`                             |
| 22*     | ReLU-norm     |          | recovery          | `normalized normalized relaxed`                                            |
| 23*     | ReLU-norm     |          | weight distance   | `--sigma {0,0.05,0.1,0.2} normalized normalized relaxed`                   |
| 24*     | ReLU-norm     |          | weight distance   | `--sigma {0,0.05,0.1,0.2} normalized normalized approx`                    |
| 25*     | ReLU-norm     |          | test error        | `--sigma {0,0.05,0.1,0.2} normalized normalized gd`                        |
| 26*     | ReLU-norm     |          | recovery          | `--k {2,3} --optw {2,3} normalized normalized relaxed`                     |
| 27*     | ReLU-norm     |          | weight distance   | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 0 normalized normalized relaxed`    |
| 28*     | ReLU-norm     |          | weight distance   | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 2 normalized normalized relaxed`    |
| 29*     | ReLU-norm     |          | weight distance   | `--sigma {0,0.05,0.1,0.2} --k 3 --optw 2 normalized normalized relaxed`    |
| 30*     | ReLU-norm     |          | test error        | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 3 normalized normalized gd`         |
| 31*     | ReLU-norm     |          | test error        | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 0 normalized normalized gd`         |
| 32*     | ReLU-norm     |          | test error        | `--sigma {0,0.05,0.1,0.2} --k 2 --optw 0 normalized normalized gd`         |

## Neural Isometry Condition figures

| Figure | Condition  | Command (after `python nic.py`) |
| 10     | NNIC-k      |

# Codebase

The codebase is structured as follows:

- `main.py`: the main CLI.
- `nic.py`: Neural Isometry Condition implementations.
- `plot.py`: for plotting data, as well as a CLI that plots results from a NumPy file containing a numpy array of shape (n, d, sample).
- `training/`: training utilities for solving the actual optimization problems.
    - `common.py`: helpers for formulating the convex optimization problems.
    - `cvx_base.py`: the abstract superclass for implementing the convex problems.
    - `cvx_normalized.py`: the implementation for the convex formulation of a ReLU network with batch normalization.
    - `cvx_skip.py`: the implementation for the convex formulation of a ReLU network with a skip connection. Can also be used for a plain network with no skip connection.
    - `noncvx_networks.py`: simple neural networks implemented in PyTorch.
    - `noncvx_network_train.py`: nonconvex neural network training code with PyTorch.

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

