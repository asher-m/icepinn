# NeuralPDE for Sea Ice
Research code for solving the forced advection-diffusion inverse problem for polar sea ice concentration.


#### Note on GitHub Markdown
On GitHub, this document renders in a way that is neither aesthetically pleasing nor easy to read.

The author's suggestion is to open this file in VSCode, with a Markdown renderer extension installed, or in another Markdown viewer, such as Obsidian. 


#### Note on This Document
While it is my hope that this summary of the methods and current work can always be complete, correct, and concise, it is the case that this page serves as a place to record ideas and thoughts on how to proceed, which, by their nature, can virtually never satisfies those desirable qualities.

If you spot any egregious errors, please open a bug with the details or your questions or a pull request with the correction.


#### Note on Notation
I've attempted to maintain consistent notation throughout.  However, in various places certain notations are more or less useful, and I possess not the foresight to use such notation throughout or it was unnecessarily cumbersome to use everywhere.  In these places, I have used the notation I thought to minimize ambiguity about the topics discussed.


## The Problem
We're trying to understand ice flows in Earth's polar regions.  NOAA and NASA publish large sea ice concentration datasets dating back nearly 50 years.

From observation, we believe the relevant physics to be governed by a forced advection-diffusion equation, like
```math
    u_t = \nabla \cdot \left( \kappa \nabla u \right) - \nabla \cdot (\vec{v}  u) + f
```
where space $x \in \mathbb{R}^2$, time $t \in \mathbb{R}^+$, the diffusion field $\kappa : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$, the velocity field $(v_1, v_2)^T =: \vec{v} : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}^2$, the forcing term $f : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ corresponding to ice freeze and thaw, and sea ice concentration $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$.

The problem may be called an *anisotropic forced advection-diffusion problem.*


## Notation
This is a particularly notation-heavy discussion, so let's establish some quantities.  This is going to seem exhaustive (and/or exhausting,) but the point here is to establish a pattern so we don't have to worry (too much) about the quantities considered.

Suppose we have the PDE,
```math
\begin{aligned}
    u_t - D[u; \lambda] &= 0 & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma.
\end{aligned}
```
To facilitate discussion of this PDE, we will adopt the following general notation: let some index set $S = \{ s \}_{s=1}^{N_S}$ represent an enumeration of all sampled points for $N_S$ total samples.  Then,
```math
\begin{aligned}
    x^s             & & \sim & & & \text{the location corresponding to the } s^{th} \text{ sample like } ( x^s_n )_{n=1}^N =: x^s \in \mathbb{R}^N \\
    t^s             & & \sim & & & \text{the time corresponding to the } s^{th} \text{ sample} \\
    u^s             & & \sim & & & u(x^s, t^s) \text{, the true (experimental) value of } u \text{ at } (x^s, t^s) \\
    \lambda^s       & & \sim & & & \lambda(x^s, t^s) \text{, the true (underlying) value of } \lambda \text{ at } (x^s, t^s) \text{ like } (\lambda_n)_{n=1}^{N_\lambda} =: \lambda^s \in \mathbb{R}^{N_\lambda},
\end{aligned}
```
and, generally for a subset $S^* \subset S$,
```math
\begin{aligned}
    x^{S^*}         & & \sim & & & \{ x^s \}_{s \in S^*} \text{, the location corresponding to the } s^{th} \text{ sample at every } s \in S^* \\
    t^{S^*}         & & \sim & & & \{ t^s \}_{s \in S^*} \text{, the time corresponding to the } s^{th} \text{ sample at every } s \in S^* \\
    u^{S^*}         & & \sim & & & \{u(x^s, t^s)\}_{s \in S^*} \text{, the true (experimental) value of } u \text{ at } (x^s, t^s) \text{ at every } s \in S^* \\
    \lambda^{S^*}   & & \sim & & & \{\lambda(x^s, t^s)\}_{s \in S^*} \text{, the true (underlying) value of } \lambda \text{ at } (x^s, t^s) \text{ at every } s \in S^*.
\end{aligned}
```
For any quantity ($u$ or $\lambda$,) I will represent its estimate with a hat, ($\hat{u}$ or $\hat{\lambda}$, respectively.)  Similarly, for any quantity, I will notate its derivative with respect to a(n arbitrary) parameter ($\chi$) with a subscript of that parameter, ($u_x := \partial_\chi u$ or $\lambda_x := \partial_\chi \lambda$,) unless the use of such notation is unclear or conflicts with other necessary subscripts. 

Finally, there are at least two important subsets of $S$,
```math
\begin{aligned}
    S^\Gamma                & & \sim & & & \text{the set of sample points in } \Gamma \\
    S^{\partial\Gamma}      & & \sim & & & \text{the set of sample points on } \partial\Gamma.
\end{aligned}
```
These will be useful when discussing boundary conditions and the enforcement of physics on the interior of the domain $\Gamma$.

For now, that's all that comes to mind for important quantities and notation, so we shall proceed with our discussion of the PINN framework.


## The Approach
### PINNs in the General Inverse Problem Context
For inverse problems, PINNs are generally constructed as solution operators for a PDE with unknown parameterization.  In particular, we learn *both* the solution and unknown parameters $\lambda$ of the (possibly nonlinear) differential operator $D$.  Then, by construction, the PINN is itself a solution to the PDE for the interior of the parabolic boundary (in the terminology of Evans, see 2e p.52.)

Suppose we have the problem,
```math
\begin{aligned}
    u_t - D[u; \lambda] &= 0 & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where $\Gamma \subset\mathbb{R}^N \times \mathbb{R}^+$ some open subset of spacetime, the solution $u : \overline{\Gamma} \rightarrow \mathbb{R}$, parameterization $\lambda : \overline{\Gamma} \rightarrow \mathbb{R}^{N_\lambda}$ is some collection of parameters, and the boundary condition $g : \partial\Gamma \rightarrow \mathbb{R}$.

Suppose we have $N_S$ sampled points $(x^S, t^S, u^S)$ representing the space, time, and value of the solution at each sampled point, and we want to find for $\hat{u}(x, t)$ and $\hat{\lambda}(x, t)$.

We approximate $u(x, t)$ and $\lambda(x, t)$ with a *physics-informed neural network* $P$ as
```math
\left[ \hat{u}(x, t), \hat{\lambda}(x, t) \right] = P(x, t).
```

Conditioned on the complexity of the network and *not* subject to additional assumptions (e.g., about the PDE or solution,) the PINN $P$ reproduces optimal approximations $\hat{u}(x, t)$ and $\hat{\lambda}(x, t)$ to $u(x, t)$ and $\lambda(x, t)$, respectively, by minimizing the loss,
```math
\begin{aligned}
    L &= \biggl\| \hat{u}^s - u^s \biggr\|_{s \in S} + \biggl\| \hat{u}_t^s -  D\left[\hat{u}^s; \hat{\lambda}^s\right] \biggr\|_{s \in S^\Gamma} \\
    &= \biggl\| \hat{u}^{S} - u^{S} \biggr\| + \biggl\| \hat{u}_t^{S^\Gamma} - D\left[\hat{u}^{S^\Gamma}; \hat{\lambda}^{S^\Gamma}\right] \biggr\|
\end{aligned}
```
where $\| \cdot \|$ is some suitable notion of norm.

A **crucial realization** is that derivatives $\hat{u}_t$ and combinations of derivatives $D[\hat{u}; \hat{\lambda}]$ can be computed by applying the chain rule for compositions of functions by autodifferentiation of the PINN.

The loss $L$ is minimized in $K$ iterations on a subset of the data $S^k \subseteq S$ using a gradient descent algorithm, (e.g., BFGS and its variants, [stochastic gradient descent](https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf), [Adam](https://arxiv.org/abs/1412.6980), etc.)  For each iteration $k$, $S^k$ is (*or can be*) randomly selected from $S$ depending on computational limitations (e.g., if the dataset is too large to fit in a computer's memory or if the complexity of the PINN is such that memory requirements for its weights do not allow simultaneous training on multiple data.)  For small problems, $S^k$ can be chosen such that $S^k = S$ for every $k$.

See [Raissi et al. 2019](https://doi.org/10.1016/j.jcp.2018.10.045) for more details, on which the above is significantly based.  The [Wikipedia page on PINNs](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) also serves as an excellent resource and encyclopedia for additional resources.


### PINNs for the Constant Scalar-Parameterized Inverse Problem with Sparse Discrete Data
Raissi et al. 2019, in addition to other interesting discussion, describes how to apply the PINN framework to problems with data collected at sparse time steps.  In particular, this can be thought of as a specialization of the preceeding discussion on PINNs in the general inverse problem context.  The following discussion covers section 4.2 of Raissi et al. 2019.

Suppose we have some collection of samples at a time $t^n$ and time $t^{n+1}$.   In the notation of the preceding section, we have $N_S = N^{n}_S + N^{n+1}_S$ samples corresponding to index set $S = S^n \cup S^{n+1}$ such that,
```math
\begin{aligned}
    t^{S^n} &= t^n \\
    t^{S^{n+1}} &= t^{n+1}.
\end{aligned}
```
Note that, of course, this requires $S^n \cap S^{n+1} = \varnothing$.

Accordingly, we have coordinates and measurements, where
```math
\begin{aligned}
    x^{S^n} & & &\text{and} & x^{S^{n+1}} & & &\text{in space } \mathbb{R}^N, & \text{and} \\
    u^{S^n} & & &\text{and} & u^{S^{n+1}} & & &\text{are scalars.}
\end{aligned}
```
Let some PDE in consideration,
```math
u_t = D[u; \lambda].
```

Raissi develops this framework assuming spatiotemporally constant differential parameterization $\lambda$.  While the discussion develops assuming spatially-constant $\lambda$ for conveience, assuming temporally-constant $\lambda$ is generally necessary to guarantee uniqueness of $\lambda$, (though it is unclear if this is both necessary and sufficient.  See [this section](#uniqueness-of) for an example of why this is the case and when non-unique $\lambda$ can be avoided.)  However, [the general framework above](#pinns-in-the-general-inverse-problem-context) theoretically (though its analysis has not yet been completed) permits extraction of temporally- and even spatio-temporally varying parameterization $\lambda$.

Recall the general form of a Runge-Kutta numerical integration scheme: momentarily, assume that we have knowledge of $u$ at each of the $q$ Runge-Kutta stages, $u^{n + c_1}, u^{n + c_2}, \dots, u^{n + c_q}$, and some value of $\lambda$.  Note that, generally speaking, these values are hidden.  An RK scheme of $q$ stages for the PDE above integrating with time step $\Delta t$ can be expressed as,
```math
\begin{aligned}
    u^{n+1} &= u^n + \Delta t \sum_{i=1}^q b_i k_i \\
    \text{where} \quad k_i &= D\left[ u^n + \Delta t \sum_{j=1}^q a_{ij} k_j; \lambda \right].
\end{aligned}
```
This is equivalent to,
```math
\begin{aligned}
    u^{n+c_i} &= u^n + \Delta t \sum_{j=1}^q a_{ij} D\left[ u^{n + c_j}; \lambda \right] \\
    u^{n+1} &= u^n + \Delta t \sum_{j=1}^q b_j D\left[ u^{n + c_j}; \lambda \right].
\end{aligned}
```
Raissi demonstrates how to rearrange the previous equations to estimate the solution $u$ at the endpoints of the interval from the predicted intermediate solutions $u^{n + c_j}$ and parameters $\lambda$ as below.  Specifically, the first equation below is the first equation above reordered, and the second equation below can be found by subtracting the equations above.
```math
\begin{aligned}
    u^n_i &= u^{n+c_i} - \Delta t \sum_{j=1}^q a_{ij} D\left[ u^{n + c_j}; \lambda \right] \\
    u^{n+1}_i &= u^{n+c_i} - \Delta t \sum_{j=1}^q (a_{ij} - b_j) D\left[ u^{n + c_j}; \lambda \right]
\end{aligned}
```
for $i = 1, \dots, q$.  Note that Raissi formulates the PDE such that $D[u; \lambda]$ has the opposite sign, which results in an opposite sign associated with the $\Delta t \sum (\cdots)$ term.  

We construct a PINN $P$ to predict the values $u^{n + c_1}, u^{n + c_2}, \dots, u^{n + c_q}$ and $\lambda$ for all $x \in x^S = x^{S^n} \cup x^{S^{n+1}}$,
```math
\left[ \hat{\lambda}, \hat{u}^{n + c_1}, \hat{u}^{n + c_2}, \dots, \hat{u}^{n + c_1} \right] = P(x^S)
```
Specfically, our PINN has yielded estimates of the solution $\hat{u}^{n + c_1}(x^{S^n}), \hat{u}^{n + c_2}(x^{S^n}), \dots, \hat{u}^{n + c_1}(x^{S^n})$, from which we estimate the solution at each $(t^{S^n}, x^{S^n})$ using the equation above for $u^{n}_i$, and $\hat{u}^{n + c_1}(x^{S^{n+1}}), \hat{u}^{n + c_2}(x^{S^{n+1}}), \dots, \hat{u}^{n + c_1}(x^{S^{n+1}})$, from which we estimate the solution at each $(t^{S^{n+1}}, x^{S^{n+1}})$ using the equation above for $u^{n+1}_i$.

Finally then, we compute a loss as some suitable norm of the errors of these estimates for the solution at each $x^{S^n}$ and $x^{S^{n+1}}$,
```math
\begin{aligned}
    L &= \sum_{i=1}^q \left[ L^{n}_i + L^{n+1}_i \right], & &\text{where} \\
    L^{n}_i &= \left\| \hat{u}^{n}_i(x^{S^n}) - u^{S^n} \right\| & &\text{and} \\
    L^{n+1}_i &= \left\| \hat{u}^{n+1}_i(x^{S^{n+1}}) - u^{S^{n+1}} \right\|.
\end{aligned}
```

Finally, it is worthwhile to comment on the choice of $q$: Raissi chooses $q$ to yield suitable (desirable) accuracy of the solution and, in fact, demonstrates that *very large* choices of $q$ are compatible with the method presented in the paper and yield improved accuracy of the solution.

A straightforward but helpful reminder is that the data on which one wishes to apply this method must satisfy the CFL condition for the PDE being studied, (and note that this has no relation to the choice of $q$.)

<!-- Add comment on how to sample data, if that makes sense, as well as expand on CFL condition. -->


#### Uniqueness of $\lambda$
As a simple example of why it is necessary to assume $\lambda$ is constant on the interval $(t^n, t^{n+1})$ is to consider the advection equation,
```math
\begin{aligned}
    u_t + \lambda u_x &= 0  & &\text{in } \mathbb{R}^+ \times \mathbb{R} \\
    u(0, x) &= u_0(x)       & &\text{on } \{ t = 0 \} \times \mathbb{R}.
\end{aligned}
```
Recall the solution to this problem is given by,
```math
u(t, x) = u_0(x - \lambda t).
```

Assume $\lambda$ is not constant on the interval $(t^n, t^{n+1})$.  Then there exist infinite time-varying parameterization $\lambda$ of the PDE.  In particular, the following two parameterizations produce data recorded at the endpoints $t^n$ and $t^{n+1}$,
```math
\begin{aligned}
    \lambda(t) &= \lambda_0 & &\text{or} & \lambda(t) &= \begin{cases} \phantom{-3} \lambda_0 & t\phantom{^n} \leq t^n \\ \phantom{-}3 \lambda_0 & t^n < t < \frac{2}{3} \Delta t + t^n \\ -3 \lambda_0 & t\phantom{^n} \geq \frac{2}{3} \Delta t + t^n \end{cases}
\end{aligned}
```
for $\Delta t = t^{n+1} - t^n$.

Exactly, this produces,
```math
\begin{aligned}
    u(t^n, x) &= u_0(x - \lambda_0 t^n) & &\text{and} & u(t^n, x) &= u_0(x - \lambda_0 t^n)
\end{aligned}
```
at endpoint $t^n$, and
```math
\begin{aligned}
    u(t^{n+1}, x) &= u_0(x - \lambda_0 t^{n+1}) & &\text{and} & u(t^{n+1}, x) &= u_0(x - \lambda_0 t^n - 3 \lambda_0 \frac{2}{3}\Delta t + 3 \lambda_0 \frac{1}{3}\Delta t) \\
                                                                      & & & & &= u_0(x - \lambda_0 t^n - \lambda_0 \Delta t) \\
                                                                      & & & & &= u_0(x - \lambda_0 t^{n + 1})

\end{aligned}
```
at endpoint $t^{n+1}$.

Clearly, then, allowing temporally varying $\lambda$ will not produce a unique $\lambda$ for every PDE.

There are some assumptions that can mitigate this problem.  For example, if we assume $\lambda$ is minimally temporally varying we again recover uniqueness.  This is, in fact, a relatively weak assumption related to how the data are sampled and is, generally, reasonable.


### PINNs for the Vector Field-Parameterized Forced Advection-Diffusion Inverse Problem
This section can be thought of a generalization of the ideas in the previous section.  Herein I discuss how I've implemented the ideas from Raissi et al. 2019 to the problem and data of forced advection-diffusion of sea ice.

Recall that we wish to extract the parameterization of the PDE,
```math
    u_t = \nabla \cdot \left( \kappa \nabla u \right) - \nabla \cdot (\vec{v}  u) + f
```
where space $x \in \mathbb{R}^2$, time $t \in \mathbb{R}^+$, the diffusion field $\kappa : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$, the velocity field $(v_1, v_2)^T =: \vec{v} : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}^2$, the forcing term $f : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ corresponding to ice freeze and thaw, and sea ice concentration $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$.  *In particular,* note that we, in fact, do not care about a solution operator.

NOAA/NSIDC provides sea ice concentration data on a rectangular spacetime grid like $u(x_j, y_k, t^n)$ where $n = 0, \dots, N_t$ times at which the concentration field is measured at $j = 1, \dots, N_x$ and $k = 1, \dots, N_y$ locations in space.

In order to establish uniqueness of the parameterization, we assume the parameters $\kappa, v_1, v_2, f$ are constant on any particular interval $(t^n, t^{n+1})$.  This assumption derives from [the discussion above on uniqueness](#uniqueness-of) of parameterization of the PDE.

We construct a PINN $P$, conditioned on a collection of solutions about the timestep in study ([more on this below](#expressivity-of-the-neural-network-and-conditioning-on-the-solution),) to yield the parameters $\kappa, v_1, v_2, f$ and the solution at each of $q$ Runge-Kutta stages.  Additionally, we extend Raissi's formulation to assume spatially-varying parameters,
```math
[\hat{\kappa}^n, \hat{v}_1^n, \hat{v}_2^n, \hat{f}^n, \hat{u}^{n+c_1}, \hat{u}^{n+c_1}, \dots, \hat{u}^{n+c_q}] = P(x_j, y_k; \{\dots, u^{n-1}, u^{n}, u^{n+1}, \dots\}).
```
where it is undestood that $\hat{\kappa}^n$, $\hat{v}_1^n$, $\hat{v}_2^n$, $\hat{f}^n$, and $\hat{u}^{n+c_1}$, $\hat{u}^{n+c_1}, \dots, \hat{u}^{n+c_q}$ are functions of the spatial coordinate as $\hat{\kappa}^n(x_j, y_k)$, $\hat{v}_1^n(x_j, y_k)$, $\hat{v}_2^n(x_j, y_k)$, $\hat{f}^n(x_j, y_k)$, $\hat{u}^{n+c_1}(x_j, y_k)$, and $\hat{u}^{n+c_1}(x_j, y_k), \dots, \hat{u}^{n+c_q}(x_j, y_k)$.

From these estimates of parameters and intermediate Runge-Kutta stages, we predict the solution at temporal endpoints of the interval, that is the quantities $u(t^n, x_j, y_k)$ and $u(t^{n+1}, x_j, y_k)$.  However, because we have measurements at dense, regular (rectangular) spacetime coordinates, we predict the solution at endpoints for all for all $i, j$.  Our loss (formulated as Raissi) becomes,
```math
\begin{aligned}
    L &= \sum_{i=1}^q \left[ L^{n}_i + L^{n+1}_i \right], & &\text{where} \\
    L^{n}_i &= \sum_{j, k} \left| \hat{u}^{n}_i(x_j, y_k) - u^{n}_i(x_j, y_k) \right|^2 & &\text{and} \\
    L^{n+1}_i &= \sum_{j, k} \left| \hat{u}^{n+1}_i(x_j, y_k) - u^{n+1}_i(x_j, y_k) \right|^2
\end{aligned}
```

In practice, we add additional terms to our loss to obtain desirable physical properties of the solution.  To assert well-posedness of this problem, we add the following loss terms (encoding necessary conditions to ensure a unique solution,)
```math
\begin{aligned}
    L_\kappa &= | \kappa | & &\text{minimality of } \kappa \\
    L_{\vec{v}} &= | v_1^2 + v_2^2 | & &\text{minimality of } \vec{v} \\
    L_f &= | f | & &\text{minimality of } f \\
    L_{\nabla \kappa} &= | \kappa_x^2 + \kappa_y^2 | & &\text{regularity of } \kappa \\
    L_{\nabla \vec{v}} &= | (v_1)_x^2 + (v_1)_y^2 + (v_2)_x^2 + (v_2)_y^2 | & &\text{regularity of } \vec{v}.
\end{aligned}
```


#### Expressivity of the Neural Network and Conditioning on the Solution
Perhaps the most significant innovation of the method I present here is conditioning the network against the known solution.  To my knowledge, the analytic tools to quantitatively understand the need to pursue this method do not exist.  Instead, I will try to qualitatively motivate this technique, novel in the PINN framework: our domain is approximately $300 \times 400 \times 20,000 \sim 10^9$ coordinate-data pairs, each with conceivably different (though likely similar) parameterization and solution to the PDE.  The expressivity, and thus scale of a network, required to capture the features of these data is significant and conceivably beyond the computational limitations of consumer hardware.  Furthermore, as our interest is primarily in extracting the parameterization of the PDE, we lose nothing by allowing the PINN to "peak" at the solution.  (For this last point, it is worth noting that we can choose the solutions we allow the PINN to see, thus regaining the ability to construct predictions, depending on the exact construction on the PINN.)  By conditioning the network on the solution, we reduce the size of the network while maintaining its expressivity and thus its ability to capture the solution and parameterization of the PDE.

In fact, we can do better: by conditioning the network on Gaussian kernels at nodes of a stencil local to a given spacetime coordinate, we further reduce the size of the network while maintaining locality and speficity of the information provided to the deep (hidden) layers of the network.  This is in contrast to the typical architecture of transformer-type networks (as ours may be classified) which typically use convolutional layers, smearing away high-frequency information on which our solution and parameters depend.

For these reasons, I present a novel method (albeits familiar to other domains) to encode the solution in a computationally-lightweight manner: convolution of the solution against a stencil of Gaussian kernels about the spacetime coordinate in study.


##### Convolution Against the Solution: An Example
The exact method is best described simply by example.

We are interested in the parameters of our PDE on an interval $(t^n, t^{n+1})$ at the location $(x_j, y_k)$.  Our solution has "inertia," so it's conceivable that we should provide both historical and future information around this interval.  We choose 6 timesteps, $\{t^{n-2}, t^{n-1}, t^{n}, t^{n+1}, t^{n+2}, t^{n+3}\}$, on which to condition our model.  For generality's sake, I'll refer to the number of timesteps on which to condition the model as the *window*.  In this example, the window has width $w_w = 6$ (the total number of timesteps in the window) and offset $w_o = -2$ (the first timestep convolved against relative to timestep $n$, respectively $t^n$.)

A [quick search](https://www.google.com/search?q=how+fast+does+an+iceberg+travel) yields that the maximum velocity of detached (mobile) ice in the ocean—that is, an iceberg—is approximately 4 km/h, and, in general, much slower.  With 25 km grid spacing, as in the NOAA/NSIDC g02202 data, we estimate the domain of dependence of our PDE to be enclosed in a five-by-five grid stencil in space.  We'll refer to the side length of this stencil as $r$, that is, $r = 5$.

Altogether, we have constructed a $w_w \times r \times r = 6 \times 5 \times 5$ stencil in spacetime, where each point corresponds to a spacetime location at which the solution was sampled.

We construct a set $C^n_{jk}$ of discrete convolutions of the solution against Gaussian kernels positioned at each spacetime coordinate in the convolution stencil, as,
```math
C^n_{jk} = \bigcup_{n'=n+w_o}^{w_w} \bigcup_{j'=j-\lfloor{r/2\rfloor}}^{\lfloor{r/2\rfloor}} \bigcup_{k'=k-\lfloor{r/2\rfloor}}^{\lfloor{r/2\rfloor}} \left( \sum_{j''=1}^{N_x} \sum_{k''=1}^{N_y} u^{n'}(x_{j''}, y_{k''}) \cdot \exp{\left[ -\frac{1}{2\sigma^2} \left( (x_{j''} - x_{j'})^2 + (y_{k''} - y_{k'})^2 \right) \right]} \right)
```
where $\sigma$ is a learnable parameter.

An obvious extension (though, as of May, 2025, I have yet to implement this,) is to convolve in time as well, as,
```math
C^n_{jk} = \bigcup_{n'=n+w_o}^{w_w} \bigcup_{j'=j-\lfloor{r/2\rfloor}}^{\lfloor{r/2\rfloor}} \bigcup_{k'=k-\lfloor{r/2\rfloor}}^{\lfloor{r/2\rfloor}} \left( \sum_{n''=0}^{N_t} \sum_{j''=1}^{N_x} \sum_{k''=1}^{N_y} u^{n''}(x_{j''}, y_{k''}) \cdot \exp{\left[ -\frac{1}{2\sigma_{xy}^2} \left( (x_{j''} - x_{j'})^2 + (y_{k''} - y_{k'})^2 \right) -\frac{1}{2\sigma_t^2} (t^{n''} - t^{n'})^2 \right]} \right)
```
where $\sigma_{xy}$ and $\sigma_t$ are learnable parameters.

In practice, we clip the exponentials used in the computation of each of these convolutions in order to avoid loading data that only minorly contributes to the solution.  This is substantiated by observing that $\sigma$ or $\sigma_{xy}$ and $\sigma_t$, respectively, train to values inside the width of one spacetime cell.


#### Returning to the Broader Problem

The set of convolutions $C_{jk}^n$ is then fed into a sufficiently-deep stack of fully-connected layers that compute the output features $\hat{\kappa}^n(x_j, y_k)$, $\hat{v}_1^n(x_j, y_k)$, $\hat{v}_2^n(x_j, y_k)$, $\hat{f}^n(x_j, y_k)$, $\hat{u}^{n+c_1}(x_j, y_k)$, and $\hat{u}^{n+c_1}(x_j, y_k), \dots, \hat{u}^{n+c_q}(x_j, y_k)$.


## Diagrams
There is no standard for diagrams (except possibly Microsoft Visio, which is itself ideologically- and cost-prohibitive,) but, for this project, I am using [draw.io](https://draw.io/) for diagrams.


## The Software
There exist a few implementations of PINNs in various ML toolkits.  
1. Raissi has [the original TensorFlow implementation](https://github.com/maziarraissi/PINNs) on Github that was used to produce the results for [the 2019 paper.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
2. [An updated repository](https://github.com/rezaakb/pinns-torch) built with PyTorch exists and provides probably a lot of the functionality I'm going to reimplement here.

We're trying something related but different enough to warrant a unique/our own implementation.

The author suggests users of this software use Anaconda/conda to maintain the python environment in which this code operates, but you should use whatever works and what you know best.

### Conda
The following commands should work with conda to create an environment called neuralpde in which this code will run.  Begin by creating the environment,
```shell
conda create -n neuralpde python=3.11 pip
```
Use conda to manage the Python version and environment itself, but install the project dependencies with `pip` after activation.  This avoids mixing compiled packages from `conda-forge` with PyPI wheels for PyTorch, which can cause OpenMP runtime conflicts on Windows.

Be sure to activate the new environment you just created.  In this tutorial, that is probably with the command `conda activate neuralpde`.

Then install the non-PyTorch dependencies with `pip`,
```shell
conda activate neuralpde
python -m pip install ipython scipy numpy matplotlib jupyter jupyterlab tqdm basemap basemap-data-hires netcdf4
```

Finally, install a suitable version of PyTorch.  This was previously possible with conda, but the PyTorch collaboration ceased its official support for the platform, so PyPI/pip is the only remaining convenient way of doing so.

Go to [pytorch.org](https://pytorch.org/), scroll down, and select a suitable version of PyTorch for your machine.  With CUDA, the command you need is most likely,
```shell
conda activate neuralpde
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Without CUDA, it is most likely,
```shell
conda activate neuralpde
python -m pip install torch torchvision torchaudio
```

#### Altogether
```shell
conda create -n neuralpde python=3.11 pip
conda activate neuralpde
python -m pip install ipython scipy numpy matplotlib jupyter jupyterlab tqdm basemap basemap-data-hires netcdf4
```
Finally, with CUDA:
```shell
conda activate neuralpde
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Without CUDA:
```shell
conda activate neuralpde
python -m pip install torch torchvision torchaudio
```

(You can omit repeated `conda activate neuralpde` commands if you're certain that you're in the right environment.)


#### Do I need `python -m pip install ...`?
Yes.  Unfortunately, on Windows in particular though possibly affecting other systems, path management can be imprecise or commands may exist in multiple locations.  `python -m pip install ...` guarantees installed packages are placed where the python used to invoke that command can find them.

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
where you install the most appropriate versions of the `torch*` packages for your system.

### Installing `neuralpde`
After setting up the environment, install the package: from the package root, if using conda run
```shell
conda activate neuralpde
python -m pip install --editable .
```
If not using conda, simply run
```shell
python -m pip install --editable .
```

If you do not wish to develop the package, or it serves no purpose for it to be installed as an editable package, simply omit `--editable`.


## Sea Ice Data
### Version 6
See [this](https://nsidc.org/data/g02202/versions/6) for more information on NOAA/NSIDC sea ice concentration data format 6.  In particular, the user manual is of significant aid.

**Note that none of the codebase has yet been updated to support V6 data.**  Work for V6 data is in progress.

#### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V6/
cd data/V6/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/aggregate/
```
