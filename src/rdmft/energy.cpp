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
double xc_energy(BasisType& basis, const arma::mat& C_AO, const arma::vec& n, double power, int n_alpha) {
  if(C_AO.n_cols == 0) return 0.0;
  
  if (n_alpha > 0 && n_alpha < (int)C_AO.n_cols) {
      // Split spin channels
      int Na = n_alpha;
      int Nb = C_AO.n_cols - Na;
      
      arma::mat Ca = C_AO.cols(0, Na - 1);
      arma::mat Cb = C_AO.cols(Na, Na + Nb - 1);
      
      arma::vec na = n.head(Na);
      arma::vec nb = n.tail(Nb);
      
      // Recursive calls for each channel (treated as single channel)
      return xc_energy(basis, Ca, na, power, 0) + xc_energy(basis, Cb, nb, power, 0);
  }

  // Single channel logic (assumed alpha or indistinguishable, or Restricted Spatial)
  arma::uword Norb = C_AO.n_cols;
  
  if (n.n_elem == 2 * Norb) {
      // Restricted Spatial Orbitals, Spin-Resolved Occupations
      arma::vec na = n.head(Norb);
      arma::vec nb = n.tail(Norb);
      return xc_energy(basis, C_AO, na, power, 0) + xc_energy(basis, C_AO, nb, power, 0);
  }
  
  if(n.n_elem != Norb) {
      // Only support 1:1 map now
      throw std::logic_error("xc_energy: occupation vector size mismatch (expected same as C cols)");
  }

  const double occ_eps = 1e-14;
  arma::vec n_eff = arma::clamp(n, 0.0, 1.0);
  arma::vec pow_n = arma::pow(arma::clamp(n_eff, occ_eps, 1.0), power);

  arma::mat P_pow = C_AO * arma::diagmat(pow_n) * C_AO.t();
  arma::mat K = basis.exchange(P_pow);
  double xc_prefactor = 0.5; // Exchange is 0.5 * Tr(P * K)
  
  return xc_prefactor * arma::trace(P_pow * K);
}

template <typename BasisType>
double compute_energy(BasisType& basis,
                      const arma::mat& Hcore,
                      const arma::mat& C_AO,
                      const arma::vec& n,
                      double power,
                      int n_alpha) {
  double E_core = core_energy<BasisType>(Hcore, C_AO, n);
  double E_J = hartree_energy<BasisType>(basis, C_AO, n);
  double E_xc = xc_energy<BasisType>(basis, C_AO, n, power, n_alpha);
  return E_core + E_J + E_xc;
}

// Explicit instantiations
template double core_energy<helfem::atomic::basis::TwoDBasis>(const arma::mat&, const arma::mat&, const arma::vec&);
template double hartree_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&);
template double xc_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, double, int);
template double compute_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, int);

} // namespace rdmft
} // namespace helfem
