#ifndef HELFEM_RDMFT_MULLER_H
#define HELFEM_RDMFT_MULLER_H

#include "muller_functional.h"
#include "rdmft_optimizer.h"

#include <armadillo>
#include <stdexcept>

namespace helfem {
namespace rdmft {

// Convenience driver for a Müller/power functional RDMFT optimization.
//
// Requirements on BasisType:
//   arma::mat overlap() const;
//   arma::mat kinetic() const;
//   arma::mat nuclear() const;
//   arma::mat coulomb(const arma::mat&) const;
//   arma::mat exchange(const arma::mat&) const;
//
// Notes:
// - Occupations are constrained to 0..1 and sum to Nelec via cosine-square + projection.
// - Natural orbitals are optimized on the generalized Stiefel manifold C^T S C = I.
// - Joint optimization is performed on the product space (Stiefel x occupations).

template <typename BasisType>
inline Result optimize_muller(BasisType& basis,
                             double Nelec,
                             size_t Norb,
                             double power,
                             arma::mat& C_AO,
                             arma::vec& n,
                             const Options& opt) {
  if (Norb == 0) throw std::logic_error("Norb must be > 0");

  arma::mat S = basis.overlap();
  arma::mat Hcore = basis.kinetic() + basis.nuclear();

  // Initial guess: generalized eigenvectors of Hcore
  arma::vec eps;
  arma::mat evec;
  bool ok = arma::eig_sym(eps, evec, Hcore, S);
  if (!ok) {
    throw std::runtime_error("Failed generalized eigensolver for initial guess (Hcore,S)");
  }
  if (evec.n_cols < Norb) throw std::logic_error("Basis too small for requested Norb");

  C_AO = evec.cols(0, Norb - 1);

  // Initial occupations: uniform in [0,1] subject to sum constraint.
  n.set_size(Norb);
  n.fill(std::min(1.0, Nelec / double(Norb)));
  n = project_capped_simplex(n, Nelec);

  // Convert to theta variables.
  arma::vec theta = arma::acos(arma::sqrt(n));

  helfem::MullerFunctional<BasisType> func(basis, Hcore, power);
  helfem::rdmft::JointOptimizer<helfem::MullerFunctional<BasisType>> optimizer(S, func, Nelec, opt);

  Result res = optimizer.minimize(C_AO, theta);

  n = occ_from_theta(theta);
  if (opt.enforce_sum_projection) {
    n = project_capped_simplex(n, Nelec);
  }

  return res;
}

} // namespace rdmft
} // namespace helfem

#endif
