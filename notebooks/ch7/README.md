# Chapter 7: Regression - Stanley H. Chan

## Summary and Implementation Guide for Data Science

This document contains a comprehensive synthesis of Chapter 7, designed for code development and deep understanding of regression phenomena.

---

## 1. Fundamental Concepts (4 Key Parts)

### Part I: The Least Squares Problem

Regression seeks a function $g_{\theta}(x)$ that approximates the unknown relationship between inputs $x_n$ and outputs $y_n$.

* **Linear Model:** Defined as $y = X\theta + e$, where $X$ is the $N \times d$ design matrix.

* **Normal Equation (Theorem 7.1):** To minimize the quadratic loss $\mathcal{E}(\theta) = \|y - X\theta\|^2$, the optimal solution $\hat{\theta}$ must satisfy:

    $$X^T X \theta = X^T y$$

* **Overdetermined Systems ($N > d$):** If $X$ is full rank, the solution is unique:

    $$\hat{\theta} = (X^T X)^{-1} X^T y$$

* **Underdetermined Systems ($N < d$):** There are infinite solutions. The **minimum-norm** solution (Theorem 7.2) is:

    $$\hat{\theta} = X^T (XX^T)^{-1} y$$

* **Basis Functions:** To model nonlinearities, inputs are transformed: $g_{\theta}(x) = \sum_{p=0}^{d-1} \theta_p \phi_p(x)$. Example: Legendre polynomials for greater numerical stability.

### Part II: Overfitting

Overfitting occurs when the model "memorizes" the training noise instead of learning the underlying pattern.

* **Training vs. Test Error:**

    * $\mathcal{E}_{train} = \sigma^2 (1 - \frac{d}{N})$ (Theorem 7.3): Decreases as model complexity $d$ increases.

    * $\mathcal{E}_{test} = \sigma^2 (1 + \frac{d}{N})$ (Theorem 7.4): Increases as model complexity $d$ increases.

* **Causes:** Few data points ($N$ small), very complex model ($d$ large), or high noise ($\sigma^2$).

### Part III: Bias-Variance Decomposition

The expected test error can be broken down into three irreducible components (Theorem 7.6):

$$\mathcal{E}_{test} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

* **Bias:** Error due to incorrect model assumptions (e.g., fitting a straight line to curved data).

* **Variance:** Sensitivity of the model to fluctuations in the training set.

* **Relationship:** Simple models have high bias/low variance; complex models have low bias/high variance.

### Part IV: Regularization (Ridge and LASSO)

To combat variance, a penalty $R(\theta)$ is added to the loss function:

1.  **Ridge Regression ($L_2$):** Adds $\lambda \|\theta\|^2$.
    * Solution: $\hat{\theta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$.
2.  **LASSO ($L_1$):** Adds $\lambda \|\theta\|_1$.
    * Property: Produces **sparse solutions**, acting as a variable selector.

---

## 2. Exercises from the Text (Full Transcript)

**Exercise 1.** (a) Construct a dataset with $N = 20$ samples, following the model $y_n = \sum_{p=0}^{d-1} \theta_p L_p(x_n) + e_n$, where $\theta_0 = 1, \theta_1 = 0.5, \theta_2 = 0.5, \theta_3 = 1.5, \theta_4 = 1$, for $-1 < x < 1$. $L_p(x)$ is the Legendre polynomial. Noise $e_n \sim \mathcal{N}(0, 0.25^2)$.

(b) Run regression with $d = 5$. Plot predicted curve and training samples.

(c) Repeat with $d = 20$. Explain observations.

(d) Increase $N$ to 50, 500, 5000 and repeat (c).

(e) Construct a testing dataset with $M = 1000$ samples and compute testing error for models in (b)-(d).

**Exercise 2.**

Consider $x_n = \sum_{k=0}^{N-1} c_k e^{-j2\pi kn/N}, n = 0, \dots, N-1$.

(a) Write in matrix-vector form $x = Wc$.

(b) Show that $W$ is orthogonal, i.e., $W^H W = I$.

(c) Using (b), derive the least squares regression solution.

**Exercise 3.**

Consider a simplified LASSO problem: $\hat{\theta} = \text{argmin}_{\theta \in \mathbb{R}^d} \|y - \theta\|^2 + \lambda \|\theta\|_1$.

Show that the solution is: $\hat{\theta} = \text{sign}(y) \cdot \max(|y| - \lambda, 0)$.

**Exercise 4.**

A 1D signal is corrupted by blur and noise: $y_n = \sum_{\ell=0}^{L-1} h_{\ell} x_{n-\ell} + e_n$.

(a) Formulate in matrix-vector form $y = Hx + e$.

(b) Consider $R(x) = \sum_{n=2}^N (x_n - x_{n-1})^2$. Show $R(x) = \|Dx\|^2$ and find $D$.

(c) Derive the regularized solution for: $\text{minimize}_x \|y - Hx\|^2 + \lambda \|Dx\|^2$.

**Exercise 5.**

Let $\sigma(a) = \frac{1}{1+e^{-a}}$.

(a) Show $\tanh(a) = 2\sigma(2a) - 1$.

(b) Show $\theta_0 + \sum \theta_p \sigma(\frac{x_n-\mu_j}{s})$ is equivalent to $\alpha_0 + \sum \alpha_p \tanh(\frac{x_n-\mu_j}{2s})$.

(c) Find the relationship between $\theta_p$ and $\alpha_p$.

**Exercise 6. (NHANES Part 1)**

(a) Derive $\hat{\theta} = \text{argmin}_{\theta} \|y - X\theta\|^2$. State conditions for unique minimum.

(b) For NHANES, label male as +1 and female as -1. Implement in Python to solve.

(c) Repeat (b) using CVXPY and compare.

**Exercise 7. (NHANES Part 2)**

Classifier: $\text{predicted label} = \text{sign}(g_{\theta}(x))$, where $g_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$.

(a) Visualize: (i) Plot male (blue circles) and female (red dots). (ii) Overlay decision boundary.

(b) Report: (i) Type 1 error, (ii) Type 2 error, (iii) Precision and Recall.

**Exercise 8. (NHANES Part 3)**

Consider: (7.48) Ridge, (7.49) Constrained $\theta$, (7.50) Constrained error.

(a) Plot $\|X\hat{\theta}_\lambda - y\|^2$ vs $\|\hat{\theta}_\lambda\|^2$, vs $\lambda$, and $\|\hat{\theta}_\lambda\|^2$ vs $\lambda$.

(b) (i) Write Lagrangians. (ii) State KKT conditions. (iii-v) Prove relationships between $\lambda, \alpha, \epsilon$.

**Exercise 9.**

Weighted Least Squares: $\hat{\theta} = \text{argmin}_{\theta} \sum_{n=1}^N w_n (y_n - x_n^T \theta)^2$.

Find the solution and discuss how to choose $w$.

**Exercise 10.**

Input noise: $x_n$ is corrupted by $e_n \sim \mathcal{N}(0, \sigma^2 I)$.

Show that $\hat{\theta} = \text{argmin}_{\theta} \sum_{n=1}^N \mathbb{E}_{e_n} [(y_n - (x_n + e_n)^T \theta)^2]$ is equivalent to Ridge regression.

---

## 3. Deep Analysis Problems (For Cursor/Programming)

Designed to foster understanding of the phenomena through simulation and code:

1.  **Simulation of the Theoretical Error Curve:** Write a script that generates multiple datasets with fixed $d$ and varying $N$. Plot the training and test error and verify if they exactly converge to $\sigma^2(1 - d/N)$ and $\sigma^2(1 + d/N)$.

2.  **Visualization of Numerical Instability:** Create a design matrix $X$ based on the Hilbert matrix. Try solving the normal equation and observe how small changes in $y$ produce large changes in $\theta$. Implement Ridge to see stabilization.

3.  **Animation of Coordinate Descent (LASSO):** Implement the optimization algorithm for LASSO. Create an animation that shows how the coefficients $\theta_j$ go to zero as you increase $\lambda$.

4.  **"Double Descent" Phenomenon:** Investigate what happens when $d$ crosses the $N$ threshold. Plot the test error for $d$ from $1$ to $3N$ using the minimum-norm solution when $d > N$.

5.  **Residual and Outlier Analysis:** Implement $L_2$ (quadratic) regression and $L_1$ (robust) regression. Introduce a massive outlier into the dataset and plot both lines to see which is more sensitive.

6.  **K-Fold Cross-Validation from Scratch:** Develop a function that splits the dataset and selects the optimal $\lambda$ for Ridge. Compare the computation time vs. the analytic formula (Leave-one-out).

7.  **Noise Propagation in Inputs:** Simulate Exercise 10. Train models with noise in $X$ and compare the resulting weights to a Ridge model trained with clean data.

8.  **Norm Geometry:** Create a 2D plot of the quadratic loss level sets and overlay the confidence regions of the $L_1$ (diamond) and $L_2$ (circle) norms. Visually identify why LASSO touches the axes.

9.  **Hyperparameter Sensitivity Analysis:** For a high-degree polynomial model, plot the actual "Bias-Variance Tradeoff" by calculating bias and variance over 1000 Monte Carlo runs.

10. **Signal Compression with LASSO:** Use a Fourier basis (sines and cosines). Use LASSO to find a sparse representation of a noisy signal and reconstruct the signal using only the nonzero coefficients.
