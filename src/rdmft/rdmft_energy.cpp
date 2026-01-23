#include "rdmft_energy.h"
#include "rdmft_gradients.h"
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
double xc_energy(BasisType& basis, const arma::mat& C_AO, const arma::vec& n, double power) {
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

template <typename BasisType>
double compute_energy_and_gradients(BasisType& basis,
                                    const arma::mat& Hcore,
                                    const arma::mat& C_AO,
                                    const arma::vec& n,
                                    double power,
                                    arma::mat& gC,
                                    arma::vec& gn) {
    // 1. Energy
    double E_core = core_energy<BasisType>(Hcore, C_AO, n);
    double E_hx = hartree_energy<BasisType>(basis, C_AO, n); 
    double E_xc = xc_energy<BasisType>(basis, C_AO, n, power);
    
    // 2. Orbital Gradients
    arma::mat gC_core;
    core_orbital_gradient(Hcore, C_AO, n, gC_core);
    
    arma::mat gC_hx;
    hartree_orbital_gradient(basis, C_AO, n, gC_hx);
    
    arma::mat gC_xc;
    xc_orbital_gradient(basis, C_AO, n, power, gC_xc);
    
    gC = gC_core + gC_hx + gC_xc;
    
    // 3. Occupation Gradients
    arma::vec gn_core;
    core_occupation_gradient(Hcore, C_AO, n, gn_core);
    
    arma::vec gn_hx;
    hartree_occupation_gradient(basis, C_AO, n, gn_hx);
    
    arma::vec gn_xc;
    xc_occupation_gradient(basis, C_AO, n, power, gn_xc);
    
    gn = gn_core + gn_hx + gn_xc;
    
    return E_core + E_hx + E_xc;
}

// Explicit instantiations
template double core_energy<helfem::atomic::basis::TwoDBasis>(const arma::mat&, const arma::mat&, const arma::vec&);
template double hartree_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&);
template double xc_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, double);
template double compute_energy_and_gradients<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, arma::mat&, arma::vec&);

} // namespace rdmft
} // namespace helfem
