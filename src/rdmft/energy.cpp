#include "energy.h"
#include "../atomic/TwoDBasis.h"
#include <stdexcept>

namespace helfem {
namespace rdmft {

template <typename BasisType>
double core_energy(const arma::mat& Hcore,
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
  // P = C * n * C^T
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  return arma::trace(P * Hcore);
}

template <typename BasisType>
double hartree_energy(BasisType& basis, const arma::mat& C_AO, const arma::vec& n) {
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

template <typename BasisType>
double compute_energy(BasisType& basis,
                      const arma::mat& Hcore,
                      const arma::mat& C_AO,
                      const arma::vec& n,
                      double power,
                      int n_alpha,
                      XCFunctionalType type) {
  double E_core = core_energy<BasisType>(Hcore, C_AO, n);
  double E_J = hartree_energy<BasisType>(basis, C_AO, n);
  double E_xc = xc_energy<BasisType>(basis, C_AO, n, type, power, n_alpha);
  return E_core + E_J + E_xc;
}

// Explicit instantiations
template double core_energy<helfem::atomic::basis::TwoDBasis>(const arma::mat&, const arma::mat&, const arma::vec&);
template double hartree_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&);
// xc_energy is instantiated in xc.cpp
template double compute_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, int, XCFunctionalType);

} // namespace rdmft
} // namespace helfem
