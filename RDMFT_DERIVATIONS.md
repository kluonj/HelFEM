# Derivation of Gradients for Power Functional

This document details the derivation of the energy gradients with respect to orbitals (coefficients) and occupation numbers for the Power (Müller) exchange-correlation functional used in this RDMFT implementation.

## 1. Definitions

### Density Matrix
The 1-reduced density matrix (1-RDM) is diagonal in the basis of natural orbitals $\{ \phi_i \}$:
$$
\gamma(\mathbf{r}, \mathbf{r}') = \sum_i n_i \phi_i(\mathbf{r}) \phi_i^*(\mathbf{r}')
$$
where $n_i \in [0,1]$ are the occupation numbers and $\sum_i n_i = N$.

### Atomic Orbital Basis
The natural orbitals are expanded in a fixed basis $\{ \chi_\mu \}$:
$$
\phi_i(\mathbf{r}) = \sum_\mu C_{\mu i} \chi_\mu(\mathbf{r})
$$
The density matrix elements in this basis are:
$$
P_{\mu\nu} = \sum_i n_i C_{\mu i} C_{\nu i}^*
$$

The symmetric matrix $\mathbf{P}$ is defined as $\mathbf{P} = \mathbf{C} \mathbf{n} \mathbf{C}^T$ where $\mathbf{n} = \text{diag}(n_1, n_2, \dots)$.

### Generalized Density Matrix
For the Power functional, we define a generalized density matrix with fractional powers of occupations:
$$
\gamma^\alpha(\mathbf{r}, \mathbf{r}') = \sum_i n_i^\alpha \phi_i(\mathbf{r}) \phi_i^*(\mathbf{r}')
$$
In the AO basis:
$$
P^\alpha_{\mu\nu} = \sum_i n_i^\alpha C_{\mu i} C_{\nu i}^*
$$
Note: For the Müller functional, $\alpha = 1/2$.

## 2. Power Functional Energy

The exchange-correlation energy is defined as:
$$
E_{xc} = - \frac{1}{2} \iint \frac{|\gamma^\alpha(\mathbf{r}, \mathbf{r}')|^2}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r} d\mathbf{r}'
$$
Substituting the expansion of $\gamma^\alpha$:
$$
E_{xc} = - \frac{1}{2} \sum_{i,j} n_i^\alpha n_j^\alpha \iint \phi_i^*(\mathbf{r}) \phi_j(\mathbf{r}) \frac{1}{|\mathbf{r}-\mathbf{r}'|} \phi_j^*(\mathbf{r}') \phi_i(\mathbf{r}') d\mathbf{r} d\mathbf{r}'
$$
Using the standard notation for exchange integrals $K_{ij} = (ij|ji) = \iint \phi_i^*(1) \phi_j(1) \frac{1}{r_{12}} \phi_j^*(2) \phi_i(2) d1 d2$:
$$
E_{xc} = - \frac{1}{2} \sum_{i,j} n_i^\alpha n_j^\alpha K_{ij}
$$

In terms of the AO basis matrices:
$$
E_{xc} = - \frac{1}{2} \text{Tr}\left( \mathbf{P}^\alpha \mathbf{K}[\mathbf{P}^\alpha] \right)
$$
where $\mathbf{K}[\mathbf{D}]$ is the exchange matrix corresponding to density $\mathbf{D}$, defined as $K_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\lambda|\nu\sigma)$.

**Code Corresondence**: The codebase implementation of `basis.exchange(P)` returns the **negative** of the exchange matrix, i.e., $-\mathbf{K}[\mathbf{P}]$. Thus, the code computes:
```cpp
E_xc = 0.5 * trace( P_alpha * basis.exchange(P_alpha) )
     = 0.5 * trace( P_alpha * (-K[P_alpha]) )
     = -0.5 * trace( P_alpha * K[P_alpha] )
```
which matches the theoretical form.

## 3. Orbital Gradient

We seek $\frac{\partial E_{xc}}{\partial C_{\mu k}}$.
Using $E_{xc} = - \frac{1}{2} \sum_{\mu\nu\lambda\sigma} P^\alpha_{\mu\nu} P^\alpha_{\lambda\sigma} (\mu\lambda|\nu\sigma)$:

Since $(\mu\lambda|\nu\sigma)$ is symmetric wrt permutation of electron indices 1 and 2 (indices pair $\mu,\nu$ and $\lambda,\sigma$), the derivative is:
$$
\frac{\partial E_{xc}}{\partial P^\alpha_{\mu\nu}} = - \sum_{\lambda\sigma} P^\alpha_{\lambda\sigma} (\mu\lambda|\nu\sigma) = - K[\mathbf{P}^\alpha]_{\mu\nu}
$$

Now apply chain rule for $P^\alpha_{\mu\nu} = \sum_k n_k^\alpha C_{\mu k} C_{\nu k}^*$:
$$
\frac{\partial P^\alpha_{\mu\nu}}{\partial C_{\rho k}} = n_k^\alpha \delta_{\mu\rho} C_{\nu k}^* + n_k^\alpha C_{\mu k} \delta_{\nu\rho}
$$
(Assuming real definition for simplicity, $C_{\nu k}^* = C_{\nu k}$):
$$
\frac{\partial E_{xc}}{\partial C_{\rho k}} = \sum_{\mu\nu} \frac{\partial E_{xc}}{\partial P^\alpha_{\mu\nu}} \frac{\partial P^\alpha_{\mu\nu}}{\partial C_{\rho k}}
$$
$$
= \sum_{\mu\nu} (-K[\mathbf{P}^\alpha]_{\mu\nu}) (n_k^\alpha \delta_{\mu\rho} C_{\nu k} + n_k^\alpha C_{\mu k} \delta_{\nu\rho})
$$
$$
= - n_k^\alpha \sum_{\nu} K_{\rho\nu} C_{\nu k} - n_k^\alpha \sum_{\mu} K_{\mu\rho} C_{\mu k}
$$
Since $\mathbf{K}$ is symmetric:
$$
\frac{\partial E_{xc}}{\partial C_{\rho k}} = - 2 n_k^\alpha \sum_{\nu} K_{\rho\nu}[\mathbf{P}^\alpha] C_{\nu k}
$$
In matrix notation:
$$
\frac{\partial E_{xc}}{\partial \mathbf{C}} = - 2 \mathbf{K}[\mathbf{P}^\alpha] \mathbf{C} \mathbf{n}^\alpha
$$

**Code Correspondence**:
The code computes:
```cpp
gC_xc = 4.0 * 0.5 * basis.exchange(Pa_pow) * C_AO * diag(pow_na)
      = 2.0 * (-K[P_alpha]) * C * n_alpha
      = - 2 * K[P_alpha] * C * n_alpha
```
This requires `basis.exchange` to return negative values (attractive potential), which matches the energy definition check.

## 4. Occupation Gradient

We seek $\frac{\partial E_{xc}}{\partial n_k}$.
$$
E_{xc} = - \frac{1}{2} \sum_{i,j} n_i^\alpha n_j^\alpha K_{ij}
$$
Differentiation:
$$
\frac{\partial E_{xc}}{\partial n_k} = - \frac{1}{2} \sum_{j} \left( \frac{\partial (n_k^\alpha)}{\partial n_k} n_j^\alpha K_{kj} + n_i^\alpha \frac{\partial (n_k^\alpha)}{\partial n_k} K_{ik} \right)
$$
(Using symmetry $K_{ij} = K_{ji}$ and swapping indices):
$$
= - \frac{1}{2} \left[ \sum_j \alpha n_k^{\alpha-1} n_j^\alpha K_{kj} + \sum_i n_i^\alpha \alpha n_k^{\alpha-1} K_{ik} \right]
$$
$$
= - \alpha n_k^{\alpha-1} \sum_j n_j^\alpha K_{kj}
$$

Let's define the matrix element in the Natural Orbital basis:
$$
\tilde{K}_{kk} = \sum_j n_j^\alpha K_{kj} = (\mathbf{C}^T \mathbf{K}[\mathbf{P}^\alpha] \mathbf{C})_{kk}
$$
Thus:
$$
\frac{\partial E_{xc}}{\partial n_k} = - \alpha n_k^{\alpha-1} (\mathbf{C}^T \mathbf{K}[\mathbf{P}^\alpha] \mathbf{C})_{kk}
$$

**Code Correspondence**:
The code loop:
```cpp
// Ka_no = C^T * basis.exchange(P_alpha) * C
//       = - C^T * K[P_alpha] * C
double val = ...;
val += 2.0 * 0.5 * power * pow(n_k, power-1) * Ka_no(k,k);
// val += alpha * n_k^{alpha-1} * (- \tilde{K}_{kk})
// val += - alpha * n_k^{alpha-1} * \tilde{K}_{kk}
```
This matches the derived gradient.
