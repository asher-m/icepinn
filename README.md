# IcePINN
Research code for solving the forced advection-diffusion inverse problem for polar sea ice concentration.


## The Problem
We're trying to understand ice floes in Earth's polar regions.  The National Oceanographic and Atmospheric Administration National Snow and Ice Data Center (NOAA/NSIDC) publish large datasets for climatologically-relevant quantities dating back about 50 years.

There exists some pretty deep, rich theory on the evolution and mechanics of sea ice.  The newest crystalized theory in this history is, broadly speaking, that of the ice thickness distribution $g(h,\, \mathbf{x},\, t)$ and its associated physical equation,
```math
g_t = - \nabla \cdot (g \, \mathbf{u}) + \Psi - \partial_h (f \, g) + L,
```
where $\mathbf{u}$ is the local advection velocity, $\Psi$ describes mechanical forces in the ice pack (e.g., convergence/compaction of the ice and divergence and separation), $f$ thermodynamic forcing, and $L$ describes lateral melt of the ice pack.

Here, we use a reduced model,
```math
c_t = - \nabla \cdot (c \, \mathbf{u}) + \nabla \cdot (\kappa \nabla c) + f
```
where the diffusion term $\kappa$ corresponds to mechanical (and diffusive) flow in the sea ice concentration, $\mathbf{u}$ corresponds to advection of the concentration, and $f$ corresponds to thermodynamic forces.


## PINN Background
I'm going to use a somewhat abstract definition of a PINN to illustrate its extension to our technique.

Raissi et al. (2019) introduce physics-informed neural networks (PINNs) as an MLP-type neural network that predicts the state variable, in our case $c$.  The key innovation of Raissi was to use automatic differentiation of the neural network to compute the PDE residual.  Thus, the thinking was that one could "trust" the results of PINN more than a purely data-driven model which made no such guarantees.

Here's what that looks like analytically: consider some PDE $c_t = D[c; \beta]$ in a domain $\Omega = \mathbf{X} \times \mathbb{R}_+$ (note that this technique generalizes beyond space-time domains, but I need to nail things down to prevent this from becoming nonsense).  Boundary data $g$ may be prescribed on some boundary $B \subset \partial\Omega$.  Data are known on the interior of the domain like $u|_\Gamma = v$ for $\Gamma \subset \Omega$ such that $\mathrm{codim}(\Gamma) > 0$.

Let some coordinate $\chi = (\mathbf{x}, t)$ such that for a neural network $\mathcal{N}$ the state variable is predicted by
```math
(c, \beta)^\top = \mathcal{N}(\chi;\, \theta)
```
for $\theta$ the parameters of the neural network.  The output $\beta$ can be considered optional: if $\beta$ is known, then the known value should be used to compute the loss and $\beta$ should not be taken as an output feature of $\mathcal{N}$; if it is unknown, then the predicted value should be used instead.

Raissi proposes (and demonstrates to impressive success) to train this network by minimizing the loss
```math
\mathcal{L} = \mathcal{L}_\text{data} + \mathcal{L}_\text{phys}
```
where
```math
\begin{aligned}
\mathcal{L}_\text{data} &= \int_\Gamma \left\| u - v \right\| \, d\mu(\mathbf{x},\, t) \\
\mathcal{L}_\text{phys} &= \int_\Phi \left\| c_t - D[c;\, \beta] \right\| \, d\nu(\mathbf{x},\, t)
\end{aligned}
```
where $\mu$ and $\nu$ are measures suitable for their respective domains (in practice, $\mu$ and $\nu$ are almost always counting measures corresponding to discrete collocation points and these integrals reduce to sums), and $\Phi$ is the domain on which to enforce physics.  Automatic differentiation of $\mathcal{N}$ is then used to compute $u_t$ and the derivatives of $D$.  This loss is minimized and the network becomes "trained".

The use of an MLP network won't be discussed significantly here: there's extensive literature that investigates the internal structure of the PINN to see if it can be optimized.  I view these as specializations of the MLP architecture that enforce certain sparcity / parameter relationships within (or even between) each layer.

On the other hand, there is work that investigates methods similar to that which I use here in which various expressions of the problem data are fed into the network, e.g., Fourier feature PINNs, some other convolutional PINN work, and more.  In this sense, the work here can be viewed as extensions of that existing work, or, regrettably and is actually the case, as having evolved in parallel to that.

Ironically, I'd argue that, if nothing else, I at least present an abstract, unified formulation of these methods by my arbitrary prescription of convolutional kernel.


## Convolutional PINN Overview
Here, I propose to use the coordinate
```math
\chi = (\mathbf{x},\, t,\, k_1 * v,\dots,\, k_N * v)
```
for the neural solution operator $\mathcal{N}$, where $k_i * v$ is a the convolution of a learned kernel $k_i$ against the data $v$.

In particular,
```math
(k_i * v)(\mathbf{x},\, t) = \int_{\mathrm{supp}(k_i)} k_i(\mathbf{x},\, t;\, \mathbf{x}',\, t') \, v(\mathbf{x} - \mathbf{x}',\, t - t') \, d\lambda(\mathbf{x})
```
for $\lambda$ the Lebesgue measure.

Once again, in practice this often reduces to a sum (e.g., if considering a discrete kernel), but not always.

Then,
```math
\begin{aligned}
    c &= \mathcal{N}(\chi;\, \theta) \\[0.5em]
    &= \mathcal{N}(\mathbf{x},\, t,\, k_1 * v,\dots,\, k_N * v).
\end{aligned}
```

The benefit of this formulation, again at least heuristically speaking, is that the neural network is able to learn the relevant space-time context (i.e., view of state variable) and the dependence of the solution on data.

In the case of the forward problem, we consider only causal kernels.  In particular, a kernel is causal if and only if,
```math
k(t,\, t') = 0 \qquad \text{for all } t' > t.
```
In the inverse problem (where the task is only parameter identification), we consider non-causal kernels.  Similarly to above,
```math
k(t,\, t') \neq 0 \qquad \text{for some } t' > t.
```

It's worth noting that it is a matter of investigation if prescribing additional kernel structure beyond its support helpful or informative.  Certain choices of $k_i$ are 1-to-1 reproductions of certain formulations of PINNs where certain features of the data are provided to the network (e.g., Fourier feature PINNs).


## Differentiable Proxy for Data
In order to evaluate the PDE residual via automatic differentiation, the solution must have a continuously differentiable dependence on the space-time coordinate $(\mathbf{x}, t)$ up to the order of the PDE.  There are circumstances when this space-time dependence will not be captured by the convolution kernel $k_i$ (i.e., discrete convolution).  Instead, we prescribe a dependence on the space-time coordinate.  In particular, we choose an interpolant that approximates the space-time variation of our data.

We *could* construct an exact interpolant, but, in general, this requires solving a system of $N$ equations, where $N$ is the number of observations.  For the sea ice problem, $N \sim 2 \times 10^9$ corresponding to $\sim 20$ GB of data.  This is too large to handle easily, so we'll work around this limitation.  In fact, even if we use a compactly supported interpolant, the resulting sparse system could still be on the order of hundreds of GB *to store in memory*.  It's worth noting that there's plenty of research that works on problems of this size (and larger), but to my knowledge routines to handle systems of this size aren't exactly "ready to go" in the sense of most other common linear algebra tools.  Instead, we construct a quasi-interpolant computed from a compact local subset of the data.

In the implementation used in this code base, this is works as follows: suppose we have discrete data $\sigma_\delta$ like
```math
\sigma_\delta: \{\mathbf{x}_i\}_{i=1}^{N_x} \times \{t_n\}_{n=1}^{N_t} \longrightarrow \mathbb{R}.
```
Essentially, we consider data on a rectangular lattice of points.

Interstitially, we define a piecewise-constant interpolant $\bar{\sigma}$ such that
```math
\bar{\sigma} : \mathrm{Convex~Hull}\left[ \{\mathbf{x}_i\}_{i=1}^{N_x} \times \{t_n\}_{n=1}^{N_t} \right] \longrightarrow \mathbb{R}
```
where
```math
\bar{\sigma}(\mathbf{x},\, t) = \sigma_\delta \left[ \argmin_i \| \mathbf{x} - \mathbf{x}_i\|,\, \argmin_n | t - t_n | \right].
```
Note that the domain of $\bar{\sigma}$ can sensibly be extended by $\Delta x / 2$ both forward and backward in each spatial dimension and $\Delta t / 2$ in time.

Finally, we construct a smooth interpolant by convolution against a tensor product cardinal cubic B-spline interpolant.  Precisely,
```math
v(\mathbf{x},\, t) = \int_{\mathrm{supp}(b)} b(\mathbf{x}' - \mathbf{x},\, t' - t)\, \bar{\sigma}(\mathbf{x}',\, \mathbf{t}') \, d\lambda(\mathbf{x}',\, t)
```
where $\lambda$ is the Lebesgue measure, and $b$ is
```math
b(\mathbf{x},\, t) = \left[ \prod_{i=1}^{\mathrm{dim}(\mathbf{X})} b_s(\mathbf{x}_i;\, \Delta x) \right] b_s(t;\, \Delta t)
```
where
```math
b_s(z;\, r) = \begin{cases}
    \dfrac{4}{3 r} - \dfrac{8 z^2}{r^3} + \dfrac{8 |z|^3}{r^4} & |z|<\dfrac{r}{2} \\[1em]
    \dfrac{8 (r - |z|)^3}{3 r^4} & \dfrac{r}{2} \le |z| < r \\[1em]
    0 & |z| \ge r.
\end{cases}
```
Note that we assume each spatial dimension has equal grid space $\Delta x$.  It is a small change to construct a method for differing grid spacing in each spatial dimension.


## Diagrams
The diagrams for this readme were created with matplotlib and [draw.io](https://draw.io/).


## Running This Software
Run the following commands from the project root.

### Conda
Create the environment and install base dependencies:
```shell
conda create -n icepinn python=3.11 pip
conda activate icepinn
python -m pip install ipython scipy numpy matplotlib jupyter jupyterlab tqdm basemap basemap-data-hires netcdf4
```

Install a suitable version of PyTorch: with CUDA,
```shell
conda activate icepinn
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Without CUDA:
```shell
conda activate icepinn
python -m pip install torch torchvision torchaudio
```

Then install icepinn as an editable package:
```shell
conda activate icepinn
python -m pip install --editable .
```

### Not Conda
Uhhh, somehow install these packages:
```
ipython
scipy
numpy
matplotlib
jupyter
jupyterlab
tqdm
basemap
basemap-data-hires
netcdf4
torch
torchvision
torchaudio
```

Then install icepinn as an editable package:
```shell
python -m pip install --editable .
```


## Sea Ice Data
### Version 6
See [this](https://nsidc.org/data/g02202/versions/6) for more information on NOAA/NSIDC sea ice concentration data format 6.  In particular, the user manual is of significant aid.

#### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V6/
cd data/V6/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/aggregate/
```

For various notebooks in this repository, you will need to prepare the data to be used.  To do so, run the notebook prepdata.ipynb.
