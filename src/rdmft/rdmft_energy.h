// rdmft_energy.h
// Defines a minimal, reusable interface for RDMFT energy functionals.

#ifndef HELFEM_RDMFT_ENERGY_H
#define HELFEM_RDMFT_ENERGY_H

#include <armadillo>

namespace helfem {
namespace rdmft {

// Minimal abstract interface for an energy functional usable by external
// code: computes energy and returns Euclidean gradients w.r.t. orbitals and
// occupations.
template <typename BasisType>
struct EnergyFunctional {
  // Evaluate energy: given AO coeffs `C` (columns = orbitals) and occupation
  // vector `n`, return energy and fill gradients `gC` and `gn`.
  virtual double energy(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn) = 0;
  virtual ~EnergyFunctional() {}
};

// -----------------------------------------------------------------------------
// Component energy helpers (merged): core, hartree (Coulomb), and XC (Muller)
// -----------------------------------------------------------------------------

// Core one-electron energy: E_core = trace(P * Hcore)
template <typename BasisType>
inline double core_energy(const arma::mat& Hcore,
                          const arma::mat& C_AO,
                          const arma::vec& n) {
  if(C_AO.n_cols == 0) return 0.0;
  arma::uword Norb = C_AO.n_cols;
  arma::vec n_sum;
  if (n.n_elem == Norb) {
    n_sum = n;
  } else if (n.n_elem == 2 * Norb) {
    n_sum = n.head(Norb) + n.tail(Norb);
  } else {
    throw std::logic_error("core_energy: occupation vector size mismatch");
  }
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  return arma::trace(P * Hcore);
}

// Hartree (Coulomb) energy: E_J = 0.5 * trace(P * J)
template <typename BasisType>
inline double hartree_energy(BasisType& basis,
                             const arma::mat& C_AO,
                             const arma::vec& n) {
  if(C_AO.n_cols == 0) return 0.0;
  arma::uword Norb = C_AO.n_cols;
  arma::vec n_sum;
  if (n.n_elem == Norb) {
    n_sum = n;
  } else if (n.n_elem == 2 * Norb) {
    n_sum = n.head(Norb) + n.tail(Norb);
  } else {
    throw std::logic_error("hartree_energy: occupation vector size mismatch");
  }
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  arma::mat J = basis.coulomb(P);
  return 0.5 * arma::trace(P * J);
}

// Müller/power XC energy helper
template <typename BasisType>
inline double xc_energy(BasisType& basis,
                               const arma::mat& C_AO,
                               const arma::vec& n,
                               double power) {
  if(C_AO.n_cols == 0) return 0.0;
  arma::uword Norb = C_AO.n_cols;
  arma::vec na, nb;
  bool split_spin = false;
  if (n.n_elem == 2 * Norb) {
    split_spin = true;
    na = n.head(Norb);
    nb = n.tail(Norb);
  } else if (n.n_elem == Norb) {
    na = n;
    nb.zeros(Norb);
  } else {
    throw std::logic_error("xc_energy: occupation vector size mismatch");
  }

  const double occ_eps = 1e-14;
  arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
  arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);
  arma::vec pow_na = arma::pow(arma::clamp(na_eff, occ_eps, 1.0), power);
  arma::vec pow_nb = split_spin ? arma::pow(arma::clamp(nb_eff, occ_eps, 1.0), power) : arma::vec();

  arma::mat Pa_pow = C_AO * arma::diagmat(pow_na) * C_AO.t();
  arma::mat Ka = basis.exchange(Pa_pow);
  double xc_prefactor = 0.5;
  double E_xca = xc_prefactor * arma::trace(Pa_pow * Ka);

  double E_xcb = 0.0;
  if(split_spin) {
    arma::mat Pb_pow = C_AO * arma::diagmat(pow_nb) * C_AO.t();
    arma::mat Kb = basis.exchange(Pb_pow);
    E_xcb = xc_prefactor * arma::trace(Pb_pow * Kb);
  }
  return E_xca + E_xcb;
}

} // namespace rdmft
} // namespace helfem

#endif // HELFEM_RDMFT_ENERGY_H
