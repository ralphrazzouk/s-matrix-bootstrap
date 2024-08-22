# Scattering Problems

## Theory

Scattering theory is a framework for studying and understanding the scattering of waves and particles. Prosaically, wave scattering corresponds to the collision and scattering of a wave with some material object, for instance (sunlight) scattered by rain drops to form a rainbow.

Scattering also includes the interaction of billiard balls on a table, the Rutherford scattering (or angle change) of alpha particles by gold nuclei, the Bragg scattering (or diffraction) of electrons and X-rays by a cluster of atoms, and the inelastic scattering of a fission fragment as it traverses a thin foil.

More precisely, scattering consists of the study of how solutions of partial differential equations, propagating freely "in the distant past", come together and interact with one another or with a boundary condition, and then propagate away "to the distant future".

The **direct scattering problem** is the problem of determining the distribution of scattered radiation/particle flux basing on the characteristics of the scatterer. The **inverse scattering problem** is the problem of determining the characteristics of an object (e.g., its shape, internal constitution) from measurement data of radiation or particles scattered from the object.

![General Interaction](/Assets/interaction.png)

In the diagram above, two particles come in with momenta $p_1$ and $p_2$, interact with some fashion, and then two particles with different momenta $p_3$ and $p_4$ leave.

## Mandelstam Variables

The Mandelstam variables are numerical quantities that encode the energy, momentum, and angles of particles in a scattering process in a Lorentz-invariant fashion. They are used for scattering processes of two particles to two particles. The Mandelstam variables were first introduced by physicist Stanley Mandelstam in 1958.

If the Minkowski metric is chosen to be $\text{diag}(1, -1, -1, -1)$, then the Mandelstam variables $s$, $t$, and $u$ are defined by
$$s = (p_1 + p_2)^2 c^2 = (p_3 + p_4)^2 c^2$$
$$t = (p_1 - p_3)^2 c^2 = (p_4 - p_2)^2 c^2$$
$$u = (p_1 - p_4)^2 c^2 = (p_3 - p_2)^2 c^2,$$
where $p_1$ and $p_2$ are the four-momenta of the incoming particles and $p_3$ and $p_4$ are the four-momenta of the outgoing particles.

$s$ is also known as the square of the center-of-mass energy (invariant mass) and $t$ as the square of the four-momentum transfer.

# $S$-matrix Theory

$S$-matrix theory was a proposal for replacing local quantum field theory as the basic principle of elementary particle physics.

It avoided the notion of space and time by replacing it with abstract mathematical properties of the $S$-matrix. In $S$-matrix theory, the $S$-matrix relates the infinite past to the infinite future in one step, without being decomposable into intermediate steps corresponding to time-slices.

This program was very influential in the 1960s, because it was a plausible substitute for quantum field theory, which was plagued with the zero interaction phenomenon at strong coupling. Applied to the strong interaction, it led to the development of string theory.

$S$-matrix theory was largely abandoned by physicists in the 1970s, as quantum chromodynamics was recognized to solve the problems of strong interactions within the framework of field theory. But in the guise of string theory, $S$-matrix theory is still a popular approach to the problem of quantum gravity.

The $S$-matrix theory is related to the holographic principle and the AdS/CFT correspondence by a flat space limit. The analog of the $S$-matrix relations in AdS space is the boundary conformal theory.

The most lasting legacy of the theory is string theory. Other notable achievements are the Froissart bound, and the prediction of the pomeron.

## History

$S$-matrix theory was proposed as a principle of particle interactions by Werner Heisenberg in 1943, following John Archibald Wheeler's 1937 introduction of the $S$-matrix.

It was developed heavily by Geoffrey Chew, Steven Frautschi, Stanley Mandelstam, Vladimir Gribov, and Tullio Regge. Some aspects of the theory were promoted by Lev Landau in the Soviet Union, and by Murray Gell-Mann in the United States.

## Basic Principles

The basic principles are

-   **Relativity**: The $S$-matrix is a representation of the Poincare group.
-   **Unitarity**: $SS^{\dagger} = 1$.
-   **Analyticity**: Integral relations and singularity conditions.

The basic analyticity principles were also called _analyticity of the first kind_, and they were never fully enumerated, but they include

1. **[Crossing](<https://en.wikipedia.org/wiki/Crossing_(physics)> "Crossing (physics)")**: The amplitudes for antiparticle scattering are the analytic continuation of particle scattering amplitudes.
2. **[Dispersion relations](https://en.wikipedia.org/wiki/Dispersion_relations "Dispersion relations")**: the values of the $S$-matrix can be calculated by integrals over internal energy variables of the imaginary part of the same values.
3. **Causality conditions**: the singularities of the $S$-matrix can only occur in ways that don't allow the future to influence the past (motivated by [Kramers–Kronig relations](https://en.wikipedia.org/wiki/Kramers%E2%80%93Kronig_relations "Kramers–Kronig relations"))
4. **Landau principle**: Any singularity of the $S$-matrix corresponds to production thresholds of physical particles.

These principles were to replace the notion of microscopic causality in field theory, the idea that field operators exist at each spacetime point, and that space-like separated operators commute with one another.

## Bootstrap Model

The basic principles were too general to apply directly, because they are satisfied automatically by any field theory. So to apply to the real world, additional principles were added.

The phenomenological way in which this was done was by taking experimental data and using the dispersion relations to compute new limits. This led to the discovery of some particles, and to successful parameterizations of the interactions of pions and nucleons.

This path was mostly abandoned because the resulting equations, devoid of any space-time interpretation, were very difficult to understand and solve.

The term "bootstrap model" is used for a class of theories that use very general consistency criteria to determine the form of a quantum theory from some assumptions on the spectrum of particles. It is a form of $S$-matrix theory.

## Regge Theory

In quantum physics, **Regge theory** is the study of the analytic properties of scattering as a function of angular momentum, where the angular momentum is not restricted to be an integer multiple of $\hbar$, but is allowed to take any complex value. The nonrelativistic theory was developed by Tullio Regge in 1959.

# Convex Optimization

The goal of convex optimization is to study the problem of **minimizing convex functions over convex sets** or, equivalently, **maximizing concave function over convex sets**. Many classes of convex optimization problems admit polynomial-time algorithms, whereas mathematical optimization is in general NP-hard.

In geometry, a subset of a Euclidean space, or more generally an affine space over the reals, is **convex** if, given any two points in the subset, the subset contains the whole line segment that joins them. Equivalently, a **convex set** or a **convex region** is a subset that intersects every line into a single line segment (possibly empty). For example, a solid cube is a convex set, but anything that is hollow or has an indent is not convex.

This is an illustration of a convex set shaped like a deformed circle.
![Convex Set](/Assets/convex_polygon.png)

This is an illustration of a non-convex set. There exist two points $x$ and $y$ such that the line connecting them does **not** completely lie inside the set.
![Non-Convex Set](/Assets/nonconvex_polygon.png)

The boundary of a convex set in the plane is always a convex curve. The intersection of all the convex sets that contain a given subset $A$ of a Euclidean space is called the convex hull of $A$. It is the smallest convex set containing $A$.

A **convex function** is a real-valued function defined on an interval with the property that its epigraph (the set of points on or above the graph of the function) is a convex set.

Convex minimization is a subfield of optimization that studies the problem of minimizing convex functions over convex sets. The branch of mathematics is devoted to the study of properties of convex sets and convex functions is called **convex analysis**.

## Definitions

**Definition:** Let $S$ be a vector space or an affine space over the real numbers, or more generally, over some ordered field (this includes Euclidean spaces, which are affine spaces). A subset $C$ of $S$ is said to be **convex** if, for all $x$ and $y$ in $C$, the line segment connecting $x$ and $y$ is included in $C$.

This means that the affine combination $(1 − t)x + ty$ belongs to $C$ for all $x$, $y$ in $C$ and $t$ in the interval $[0, 1]$. This implies that convexity is invariant under affine transformations. Further, it implies that a convex set in a real or complex topological vector space is path-connected (and therefore also connected).

A set $C$ is **strictly convex** if every point on the line segment connecting $x$ and $y$ other than the endpoints is inside the topological interior of $C$. A closed convex subset is strictly convex if and only if every one of its boundary points is an extreme point.

A set $C$ is **absolutely convex** if it is convex and balanced.

## Abstract Form

A convex optimization problem is defined by two ingredients:

-   The **objective function** $f$, which is a real-valued convex function of $n$ variables, given by $f: \mathcal{D} \subseteq \mathbb{R}^n \rightarrow \mathbb{R}$.
-   The **feasible set**, which is a convex subset $C \subseteq \mathbb{R}^n$.

The goal of the problem is to find some $\mathbf{x}^{\ast} \in C$ attaining $\text{inf} \left\{ f(\mathbf{x}) : \mathbf{x} \in C \right\}$.

In general, there are three options regarding the existence of a solution:

-   If such a point $\mathbf{x}^{\ast}$ exists, it is referred to as an **optimal point** or **solution**. The set of all optimal points is called the **optimal set** and the problem is called **solvable**.
-   If $f$ is unbounded from below over $C$, or the infimum is not attained, then the optimization problem is said to be **unbounded**.
-   Otherwise, if $C$ is the empty set, then the problem is said to be **infeasible**.

## Standard Form

A convex optimization problem is in the **standard form** if it is written as

$$
\begin{align}
	&\underset{\mathbf{x}}{\text{minimize}}& & f(\mathbf{x}) \\
	&\text{subject\ to}
	& &g_i(\mathbf{x}) \leq 0, \quad i = 1, \dots, m \\
	&&&h_i(\mathbf{x}) = 0, \quad i = 1, \dots, p,
\end{align}
$$

where

-   $\mathbf{x} \in \mathbb{R}^n$ is the vector of optimization variables
-   The objective function $f: \mathcal{D} \subseteq \mathbb{R}^n \rightarrow \mathbb{R}$ is a convex function
-   The inequality constraints $g_i: \mathbb{R}^n \rightarrow \mathbb{R}$, $i = 1, \ldots, m$, are convex functions
-   The inequality constraints $h_i: \mathbb{R}^n \rightarrow \mathbb{R}$, $i = 1, \ldots, p$, are affine transformations, that is, of the form $h_i(\mathbf{x}) = \mathbf{a}_i \cdot \mathbf{x} - b_i$, where $\mathbf{a}_i$ is a vector and $b_i$ is a scalar.

The feasible set $C$ of the optimization problem consists of all points $\mathbf{x} \in \mathcal{D}$ satisfying the inequality and the equality constraints. This set is convex because $\mathcal{D}$ is convex (the sublevel sets of convex functions are convex, affine sets are convex, and the intersection of convex sets is convex).

Many optimization problems can be equivalently formulated in this standard form. For example, the problem of maximizing a concave function $f$ can be re-formulated equivalently as the problem of minimizing the convex function $-f$. The problem of maximizing a concave function over a convex set is commonly called a convex optimization problem.

# Tools

To do convex optimization, there are multiple Python libraries that we can use, which include `pulp`, `cvxpy`, `cvxopt`, and `scipy.optimize`. We will use all libraries to cross-check our results.

-   Other tools: Gurobi, Cplex, HiGHS (problem has to be in standard form).
-   One could write a custom algorithm, which is hard, but can be faster.
-   One could use a modeling language, which is intuitive, and it translates the problem to standard form.

## Solvers

-   **ECOS** (Domahidi)
    -   cone solver
    -   interior-point method
    -   compact, library-free C code
-   **SCS** (O'Donoghue)
    -   cone solver
    -   first-order method
    -   parallelism with OpenMP
    -   GPU support
-   **OSQP** (Stellato, Banjac, Goulart)
    -   first-order method
    -   targets QPs and LPs
    -   code generation support
-   Others: **CVXOPT**, **GLPK**, **MOSEK**, **GUROBI**, **Cbc**, etc.
-   **CVXPY** is not a solver.

## Pulp

An example of using `pulp` is as follows

```python
import pulp as p

lp_prob = p.LpProblem("Optimization with pulp", p.LpMinimize)
x = p.LpVariable('x', lowBound = 0)
y = p.LpVariable('y', lowBound = 0)

lp_prob += 3*x + 4*y
lp_prob += 2*x + 3*y >= 12
lp_prob += -x + y <= 3
lp_prob += x >= 4
lp_prob += y <= 3

lp_prob # to view the setup

lp_prob.solve()
p.LpStatus[lp_prob.solve()]

p.value(x)
p.value(y)
p.value(lp_prob.objective)
```

Notice that `lowBound = 0` for `x` is redundant in this case because we already included the lower bound as a constraint.

The optimal solutions are `x = 6` and `y = 0`.

Pulp can only solve linear optimization problems.

## CVXPY

In `cvxpy`, each expression has a curvature: constant, affine, convex, concave, etc. Expressions are formulated from variables, constants, and operations. Operations are defined by `cvxpy`. The curvature of the output can be inferred from the curvature of the input.

CVXPY relies on the open source solvers [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs), [OSQP](https://osqp.org/), [SCS](http://github.com/cvxgrp/scs), and [ECOS](http://github.com/ifa-ethz/ecos). Additional solvers are supported, but must be installed separately.

```python
from cvxpy.settings import (
    CBC as CBC,
    CLARABEL as CLARABEL,
    COPT as COPT,
    CPLEX as CPLEX,
    CPP_CANON_BACKEND as CPP_CANON_BACKEND,
    CVXOPT as CVXOPT,
    DIFFCP as DIFFCP,
    ECOS as ECOS,
    ECOS_BB as ECOS_BB,
    GLOP as GLOP,
    GLPK as GLPK,
    GLPK_MI as GLPK_MI,
    GUROBI as GUROBI,
    INFEASIBLE as INFEASIBLE,
    INFEASIBLE_INACCURATE as INFEASIBLE_INACCURATE,
    MOSEK as MOSEK,
    NAG as NAG,
    OPTIMAL as OPTIMAL,
    OPTIMAL_INACCURATE as OPTIMAL_INACCURATE,
    OSQP as OSQP,
    DAQP as DAQP,
    PDLP as PDLP,
    PIQP as PIQP,
    PROXQP as PROXQP,
    ROBUST_KKTSOLVER as ROBUST_KKTSOLVER,
    RUST_CANON_BACKEND as RUST_CANON_BACKEND,
    SCIP as SCIP,
    SCIPY as SCIPY,
    SCIPY_CANON_BACKEND as SCIPY_CANON_BACKEND,
    SCS as SCS,
    SDPA as SDPA,
    SOLVER_ERROR as SOLVER_ERROR,
    UNBOUNDED as UNBOUNDED,
    UNBOUNDED_INACCURATE as UNBOUNDED_INACCURATE,
    USER_LIMIT as USER_LIMIT,
    XPRESS as XPRESS,
    get_num_threads as get_num_threads,
    set_num_threads as set_num_threads,
)
```

For a guided tour of CVXPY, check out the [tutorial](https://www.cvxpy.org/tutorial/index.html). For applications to machine learning, control, finance, and more, browse the [library of examples](https://www.cvxpy.org/examples/index.html). For background on convex optimization, see the book [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf/) by Boyd and Vandenberghe.

An example of using `cvxpy` is as follows

```python
import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

objective = cp.Minimize((x - y)**2)
constraints = [
	x + 2*y == 2,
	x - y >= 6
	]

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(x.value)
print(y.value)
print(constraints[0].dual_value)
```

This is a non-linear optimization problem, which can be solved in `cvxpy`, unlike `pulp`.
The optimal solutions are `x = 4.6666667` and `y = -1.3333333`.

### Changing the problem

[`Problems`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem "cvxpy.problems.problem.Problem") are immutable, meaning they cannot be changed after they are created. To change the objective or constraints, create a new problem.

### Other problem statuses

If a problem is infeasible or unbounded, the status field will be set to “infeasible” or “unbounded”, respectively. The value fields of the problem variables are not updated.

Notice that for a minimization problem the optimal value is `inf` if infeasible and `-inf` if unbounded. For maximization problems the opposite is true.

If the solver called by CVXPY solves the problem but to a lower accuracy than desired, the problem status indicates the lower accuracy achieved. The statuses indicating lower accuracy are

-   “optimal_inaccurate"
-   “unbounded_inaccurate”
-   “infeasible_inaccurate”

The problem variables are updated as usual for the type of solution found (i.e., optimal, unbounded, or infeasible).

If the solver completely fails to solve the problem, CVXPY throws a `SolverError` exception. If this happens you should try using other solvers. See the discussion of [Solvers](https://www.cvxpy.org/resources/related_projects/index.html#solvers) for details.

CVXPY provides the following constants as aliases for the different status strings:

-   `OPTIMAL`
-   `INFEASIBLE`
-   `UNBOUNDED`
-   `OPTIMAL_INACCURATE`
-   `INFEASIBLE_INACCURATE`
-   `UNBOUNDED_INACCURATE`
-   `INFEASIBLE_OR_UNBOUNDED`

To test if a problem was solved successfully, you would use

```
prob.status == OPTIMAL
```

The status `INFEASIBLE_OR_UNBOUNDED` is rare. It’s used when a solver was able to determine that the problem was either infeasible or unbounded, but could not tell which. You can determine the precise status by re-solving the problem where you set the objective function to a constant (e.g., `objective = cp.Minimize(0)`). If the new problem is solved with status code `INFEASIBLE_OR_UNBOUNDED` then the original problem was infeasible. If the new problem is solved with status `OPTIMAL` then the original problem was unbounded.

### Vectors and matrices

[`Variables`](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#cvxpy.expressions.variable.Variable "cvxpy.expressions.variable.Variable") can be scalars, vectors, or matrices, meaning they are 0, 1, or 2 dimensional.

```python
# A scalar variable.
a = cp.Variable()

# Vector variable with shape (5,).
x = cp.Variable(5)

# Matrix variable with shape (5, 1).
x = cp.Variable((5, 1))

# Matrix variable with shape (4, 7).
A = cp.Variable((4, 7))
```

You can use your numeric library of choice to construct matrix and vector constants. For instance, if `x` is a CVXPY Variable in the expression `A @ x + b`, `A` and `b` could be Numpy ndarrays, SciPy sparse matrices, etc. `A` and `b` could even be different types.

Currently the following types may be used as constants:

-   NumPy ndarrays
-   NumPy matrices
-   SciPy sparse matrices

### Constraints

As shown in the example code, you can use `==`, `<=`, and `>=` to construct constraints in CVXPY. Equality and inequality constraints are elementwise, whether they involve scalars, vectors, or matrices. For example, together the constraints `0 <= x` and `x <= 1` mean that every entry of `x` is between 0 and 1.

If you want matrix inequalities that represent semi-definite cone constraints, see [Semidefinite matrices](https://www.cvxpy.org/tutorial/constraints/index.html#semidefinite). The section explains how to express a semi-definite cone inequality.

You cannot construct inequalities with `<` and `>`. Strict inequalities don’t make sense in a real world setting. Also, you cannot chain constraints together, e.g., `0 <= x <= 1` or `x == y == 2`. The Python interpreter treats chained constraints in such a way that CVXPY cannot capture them. CVXPY will raise an exception if you write a chained constraint.

### Parameters

[`Parameters`](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#cvxpy.expressions.constants.parameter.Parameter "cvxpy.expressions.constants.parameter.Parameter") are symbolic representations of constants. The purpose of parameters is to change the value of a constant in a problem without reconstructing the entire problem. In many cases, solving a parametrized program multiple times can be substantially faster than repeatedly solving a new problem: after reading this section, be sure to read the tutorial on [Disciplined Parametrized Programming](https://www.cvxpy.org/tutorial/dpp/index.html#dpp) (DPP).

Parameters can be vectors or matrices, just like variables. When you create a parameter you have the option of specifying attributes such as the sign of the parameter’s entries, whether the parameter is symmetric, etc. These attributes are used in [Disciplined Convex Programming](https://www.cvxpy.org/tutorial/dcp/index.html#dcp) and are unknown unless specified. Parameters can be assigned a constant value any time after they are created. The constant value must have the same dimensions and attributes as those specified when the parameter was created.

```python
# Positive scalar parameter.
m = cp.Parameter(nonneg=True)

# Column vector parameter with unknown sign (by default).
c = cp.Parameter(5)

# Matrix parameter with negative entries.
G = cp.Parameter((4, 7), nonpos=True)

# Assigns a constant value to G.
G.value = -numpy.ones((4, 7))
```

You can initialize a parameter with a value. The following code segments are equivalent:

```python
# Create parameter, then assign value.
rho = cp.Parameter(nonneg=True)
rho.value = 2

# Initialize parameter with a value.
rho = cp.Parameter(nonneg=True, value=2)
```

Computing trade-off curves is a common use of parameters. The example below computes a trade-off curve for a LASSO problem.

```python
import cvxpy as cp
import numpy
import matplotlib.pyplot as plt

# Problem data.
n = 15
m = 10
numpy.random.seed(1)
A = numpy.random.randn(n, m)
b = numpy.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cp.Parameter(nonneg=True)

# Construct the problem.
x = cp.Variable(m)
error = cp.sum_squares(A @ x - b)
obj = cp.Minimize(error + gamma*cp.norm(x, 1))
prob = cp.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = numpy.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(x, 1).value)
    x_values.append(x.value)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(6,10))

# Plot trade-off curve.
plt.subplot(211)
plt.plot(l1_penalty, sq_penalty)
plt.xlabel(r'\|x\|_1', fontsize=16)
plt.ylabel(r'\|Ax-b\|^2', fontsize=16)
plt.title('Trade-Off Curve for LASSO', fontsize=16)

# Plot entries of x vs. gamma.
plt.subplot(212)
for i in range(m):
    plt.plot(gamma_vals, [xi[i] for xi in x_values])
plt.xlabel(r'\gamma', fontsize=16)
plt.ylabel(r'x_{i}', fontsize=16)
plt.xscale('log')
plt.title(r'\text{Entries of x vs. }\gamma', fontsize=16)

plt.tight_layout()
plt.show()
```

Trade-off curves can easily be computed in parallel. The code below computes in parallel the optimal x for each 𝛾 in the LASSO problem above.

```python
from multiprocessing import Pool

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = prob.solve()
    return x.value

# Parallel computation (set to 1 process here).
pool = Pool(processes = 1)
x_values = pool.map(get_x, gamma_vals)
```

# $S$-matrix Bootstrap

The $S$-matrix bootstrap maps out the space of $S$-matrices that are allowed by analyticity, crossing, unitarity, and other constraints. For the $2 \rightarrow 2$ scattering matrix $S_{2 \rightarrow 2}$, such space is an infinite dimensional convex space whose boundary can be determined by maximing linear functionals. On the boundary, interesting theories can be found, many times at vertices of the space.

Here, we consider $(3 + 1)$-dimensional theories and focus on the equivalent dual convex minimization problem that provides strict upper bounds for the regularized primal problem and has interesting practical and physical advantages over the primal problem. Its variables are dual partial waves $k_{\ell}(s)$ that are free variables, namely they do not have to obey any crossing, unitarity, or other constraints. Nevertheless, they are directly related to the partial waves $f_{\ell}(s)$, for which all crossing, unitarity, and symmetry properties result from the minimization. Numericallt, it requires only a few dual partial waves, much as one wants to possibly match experimental results. We consider the case of scalar fields whch is related to pion physics.

The simplest example is single pion scattering, namely a scalar particle of mass $m^2$ that we set to $m^2 = 1$ and $c = 1$. The amplitude for the process
$$\pi(p_1) + \pi(p_2) \rightarrow \pi(p_3) + \pi(p_4),$$
is a function of the Mandelstam variables
$$s = (p_1 + p_2)^2 = (p_3 + p_4)^2$$
$$t = (p_1 - p_3)^2 = (p_4 - p_2)^2$$
$$u = (p_1 - p_4)^2 = (p_3 - p_2)^2,$$
with $s + t + u = 4$. Its analytic crossing properties are captured by the Mandelstam representation that can be written as
$$F(s, t, u) = f_0 + \int_4^\infty \mathcal{K}(s, t, u; x) \, \sigma(x) ~ \mathrm{d}x + \int_4^\infty \int_4^\infty \mathcal{K}(s, t, u; x, y) \, \rho(x, y) ~ \mathrm{d}y ~ \mathrm{d}x$$
with the kernels

$$
\mathcal{K}(s, t, u; x) = \frac{1}{\pi} \left[ \frac{1}{x - s} + \frac{1}{x - t} + \frac{1}{x - u} \right]
$$

and

$$
\mathcal{K}(s, t, u; x, y) = \frac{1}{2\pi^2} \left[ \frac{1}{(x - s)(y - t)} + \frac{1}{(x - s)(y - u)} + \frac{1}{(x - u)(y - t)} \right] + (x \leftrightarrow y)
$$

The amplitude $F(s, t, u)$ has double jumps on the Mandelstam regions depicted in red in Fig. 2. The amplitude is crossing symmetric under $s \leftrightarrow t \leftrightarrow u$, and only the symmetric part of $\rho$ contributes so we can take $\rho(x, y) = \rho(y, x)$. The physical partial waves $f_{\ell} (s)$ are defined as

$$
f_{\ell}(s) = \frac{1}{4} \int_{-1}^{1} P_{\ell}(\mu) F(s^+, t, u) ~ \mathrm{d} \mu
$$

which are non-zero only for even $\ell$.

## Primal Problem

# Scratch

## Check Later

```python
from pandas.tools.plotting import scatter_matrix
```

## Portfolio Optimization

The problem is as follows

$$
\begin{align}
	& \text{maximize} & & \mu^T w - \gamma w^T \sum w \\
	& \text{subject\ to} & & \mathbf{1}^T w = 1, \quad w \in \mathcal{W}
\end{align}
$$

-   variable $w \in \mathbb{R}^n$ is the **portfolio allocation vector**
-   $\mathcal{W}$ is the **set of allowed portfolios**
-   common case: $\mathcal{W} = \mathbb{R}^n_+$ (long portfolios only)
-   $\gamma > 0$ is the **risk aversion parameter**
-   $\mu^T w - \gamma w^T \sum w$ is the **risk-adjusted return**
-   varying $\gamma$ gives the optimal **risk-return trade-off**

## CVXPYgen - Code Generation with CVXPY

CVXPYgen takes a convex optimization problem family modeled with CVXPY and generates a custom solver implementation in C. This generated solver is specific to the problem family and accepts different parameter values. In particular, this solver is suitable for deployment on embedded systems. In addition, CVXPYgen creates a Python wrapper for prototyping and desktop (non-embedded) applications.

https://github.com/cvxgrp/cvxpygen

## Scattering Kinematics (Lorentz Invariant)

We have the two Mandelstam variables because it is a $2 \rightarrow 2$ scattering problem.
$$s = - (p_1 + p_2)^2$$
$$t = - (p_1 - p_3)^3$$
$$\cos(\theta) = 1 + \frac{2t}{s - 4m^2}$$

Partial wave equation is an amplitude $M$ which is a function of two Mandelstam variables. It is an expansion using Legendre polynomials $P_\ell(\cos(\theta))$

$$
M(s, t) = \sum_{\ell = 0}^\infty (2\ell + 1) f_{\ell}(s) P_{\ell} (\cos(\theta))
$$

subject to a unitarity constraint given by
$$\left| f_ell(s) \right|^2 \leq \text{Im} \left( f_\ell(s) \right) \leq 1.$$

### Crossing: Take 1

Consider $M$ to be symmetric, _i.e._ $M(s, t) = M(t, s)$. Then, replacing $\cos(\theta)$ in $M$ and expanding we get

$$
\sum_{\ell = 0}^\infty (2\ell + 1) f_{\ell}(s) P_{\ell} \left( 1 + \frac{2t}{s - 4m^2} \right) = \sum_{\ell = 0}^\infty (2\ell + 1) f_{\ell}(t) P_{\ell} \left( 1 + \frac{2s}{t - 4m^2} \right).
$$

This is hard to solve. The Legendre polynomial partial wave for particular spin: polynomial in $t$ on one side changes to a polynomial in $s$ under crossing. Thus, the pole on one side will require an infinite number of partial waves on the other.

### Crossing: Take 2

Instead, we can redefine our variables (conformal transformation). Define
$$\rho_s = \frac{\sqrt{4m^2 - s_0} - \sqrt{4m^2 - s}}{\sqrt{4m^2 - s_0} + \sqrt{4m^2 - s}},$$
with
$$s_0 = \frac{4}{3}.$$
This gives us

$$
M(s, t) = poles + \sum_{a, b, c} \alpha_{abc} \rho_s^a \rho_t^b \rho_u^c.
$$

This form in manifestly (by definition?) crossing symmetric. There will be a need to check for unitarity $\left| f_\ell(s) \right|^2 \leq 1$. These conditions can be recast into a semi-definite programming problem, inspired by CFT bootstrap.

We want to constrain $\alpha_{abc}$ from the following. ... Adler zeros.

Fix a gauge with an ansatz.

$$
M(s, t) = \sum_{a, b} \alpha_{ab} \left( \rho_s^a \rho_t^b + \rho_s^a \rho_u^b + \rho_t^a \rho_u^b \right).
$$

This has a double discontinuity, where the previous one had a triple discontinuity.

For unitarity, we define

$$
S_\ell(s) = 1 + \frac{i}{32\pi} \sqrt{\frac{s - 4}{s}} \int_{-1}^1 P_\ell(z) M(s, z) \mathrm{d} z,
$$

where
$$z = \cos(\theta)$$
$$t = - \frac{1}{2}(s - 4)(1 - z)$$
$$u = - \frac{1}{2} (s - 4)(1 + z)$$
