#include "rdmft_optimizer.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <iostream>

// Minimal self-contained test of the joint product-manifold optimizer:
// - Generalized Stiefel for C (C^T S C = I)
// - Occupations n_i in [0,1] via n_i = cos^2(theta_i)
// - Constraint sum(n)=N enforced via augmented Lagrangian
//
// Note: This test validates the *optimizer mechanics* with a smooth objective.
// To run actual RDMFT, plug in a physical functional (e.g. MullerFunctional)
// together with a HelFEM basis (overlap + Coulomb + exchange).

namespace {

struct ToyBasis {
  arma::mat S;
};

// Smooth, well-conditioned functional compatible with helfem::rdmft::JointOptimizer.
// E(C,n) = Tr(P H) + gamma * ||n||^2, where P = C diag(n) C^T.
struct QuadraticFunctional {
  arma::mat H;
  double gamma;

  double operator()(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn) {
    arma::mat P = C * arma::diagmat(n) * C.t();
    double E = arma::trace(P * H) + gamma * arma::dot(n, n);

    // gC = 2 H C diag(n)
    gC = 2.0 * H * C * arma::diagmat(n);

    // gn_i = (C^T H C)_{ii} + 2 gamma n_i
    arma::mat Hno = C.t() * H * C;
    gn = Hno.diag() + 2.0 * gamma * n;
    return E;
  }
};

arma::mat random_spd(size_t n, unsigned seed) {
  arma::arma_rng::set_seed(seed);
  arma::mat A = arma::randn(n, n);
  arma::mat M = A.t() * A;
  // Normalize scale so eigenvalues are O(1)
  M /= (arma::trace(M) / double(n));
  M += 0.1 * arma::eye(n, n);
  return M;
}

} // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const size_t Nbf = 20;
  const size_t Norb = 8;
  const double Nelec = 4.0; // sum of occupations

  ToyBasis basis;
  basis.S = random_spd(Nbf, 1);

  arma::mat H = random_spd(Nbf, 2);
  H = 0.5 * (H + H.t());
  H *= 0.1;

  QuadraticFunctional func{H, 0.01};

  // Initial C: random then S-orthonormalized by optimizer
  arma::arma_rng::set_seed(3);
  arma::mat C0 = arma::randn(Nbf, Norb);

  // Initial theta: choose roughly Nelec by setting cos^2 average
  arma::vec theta0(Norb);
  theta0.fill(std::acos(std::sqrt(std::min(0.999, Nelec / double(Norb)))));

  helfem::rdmft::Options opt;
  opt.algorithm = helfem::rdmft::Algorithm::ConjugateGradient;
  opt.max_iter = 800;
  opt.grad_tol = 0.1;
  opt.step0 = 0.5;

  helfem::rdmft::JointOptimizer<QuadraticFunctional> optimizer(basis.S, func, Nelec, opt);

  arma::mat C = C0;
  arma::vec theta = theta0;
  auto res = optimizer.minimize(C, theta);

  arma::vec n = helfem::rdmft::occ_from_theta(theta);

  std::cout << "RDMFT test (toy)\n";
  std::cout << "iters=" << res.iters << " converged=" << res.converged << "\n";
  std::cout << "E=" << res.energy << "\n";
  std::cout << "|grad|=" << res.grad_norm << "\n";
  std::cout << "sum(n)=" << arma::sum(n) << " (target " << Nelec << ")\n";
  std::cout << "min(n)=" << n.min() << " max(n)=" << n.max() << "\n";
  std::cout << "orth_err=" << helfem::manifold::orthonormality_error(C, basis.S) << "\n";

  if (!res.converged) return 2;
  if (std::abs(arma::sum(n) - Nelec) > 1e-6) return 3;
  if (helfem::manifold::orthonormality_error(C, basis.S) > 1e-8) return 4;
  if (n.min() < -1e-12 || n.max() > 1.0 + 1e-12) return 5;

  return 0;
}
