// rdmft_gradients.h
// Small helpers for gradients and occupation transformations used by optimizers.

#ifndef HELFEM_RDMFT_GRADIENTS_H
#define HELFEM_RDMFT_GRADIENTS_H

#include <armadillo>

namespace helfem {
namespace rdmft {

arma::vec occ_from_theta(const arma::vec& theta);
arma::vec d_occ_d_theta(const arma::vec& theta);

// Project onto capped simplex 0<=x<=1, sum(x)=target_sum
arma::vec project_capped_simplex(const arma::vec& y, double target_sum, double* lambda_out = nullptr);

// -----------------------------------------------------------------------------
// Component gradient helpers (core, hartree, Muller XC)
// -----------------------------------------------------------------------------

// Core orbital gradient
void core_orbital_gradient(const arma::mat& Hcore,
                                  const arma::mat& C_AO,
                                  const arma::vec& n,
                                  arma::mat& gC_out);

// Core occupation gradient
void core_occupation_gradient(const arma::mat& Hcore,
                                    const arma::mat& C_AO,
                                    const arma::vec& n,
                                    arma::vec& gn_out);

// Hartree orbital gradient
template <typename BasisType>
inline void hartree_orbital_gradient(BasisType& basis,
                                     const arma::mat& C_AO,
                                     const arma::vec& n,
                                     arma::mat& gC_out) {
  if(C_AO.n_cols == 0) { gC_out.reset(); return; }
  arma::uword Norb = C_AO.n_cols;
  arma::vec n_sum;
  if (n.n_elem == Norb) {
    n_sum = n;
  } else if (n.n_elem == 2 * Norb) {
    n_sum = n.head(Norb) + n.tail(Norb);
  } else {
    throw std::logic_error("hartree_orbital_gradient: occupation vector size mismatch");
  }
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  arma::mat J = basis.coulomb(P);
  gC_out = 2.0 * J * C_AO * arma::diagmat(n_sum);
}

// Hartree occupation gradient
template <typename BasisType>
inline void hartree_occupation_gradient(BasisType& basis,
                                       const arma::mat& C_AO,
                                       const arma::vec& n,
                                       arma::vec& gn_out) {
  if(C_AO.n_cols == 0) { gn_out.reset(); return; }
  arma::uword Norb = C_AO.n_cols;
  arma::vec n_sum;
  if (n.n_elem == Norb) {
    n_sum = n;
  } else if (n.n_elem == 2 * Norb) {
    n_sum = n.head(Norb) + n.tail(Norb);
  } else {
    throw std::logic_error("hartree_occupation_gradient: occupation vector size mismatch");
  }
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  arma::mat J = basis.coulomb(P);
  arma::mat J_no = C_AO.t() * J * C_AO;
  arma::uword Nocc = n.n_elem;
  gn_out.set_size(Nocc);
  if (Nocc == Norb) {
    for(arma::uword i=0;i<Norb;++i) gn_out(i) = J_no(i,i);
  } else {
    for(arma::uword i=0;i<Norb;++i) { gn_out(i) = J_no(i,i); gn_out(Norb + i) = J_no(i,i); }
  }
}

// XC orbital gradient (Müller/power)
template <typename BasisType>
inline void muller_xc_orbital_gradient(BasisType& basis,
                                       const arma::mat& C_AO,
                                       const arma::vec& n,
                                       double power,
                                       arma::mat& gC_out) {
  if(C_AO.n_cols == 0) { gC_out.reset(); return; }
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
    throw std::logic_error("muller_xc_orbital_gradient: occupation vector size mismatch");
  }

  const double occ_eps = 1e-14;
  arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
  arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);
  arma::vec pow_na = arma::pow(arma::clamp(na_eff, occ_eps, 1.0), power);
  arma::vec pow_nb = split_spin ? arma::pow(arma::clamp(nb_eff, occ_eps, 1.0), power) : arma::vec();

  arma::mat Pa_pow = C_AO * arma::diagmat(pow_na) * C_AO.t();
  arma::mat Ka = basis.exchange(Pa_pow);
  double xc_prefactor = 0.5;
  arma::mat gC_xc = 4.0 * xc_prefactor * Ka * C_AO * arma::diagmat(pow_na);
  if(split_spin) {
    arma::mat Pb_pow = C_AO * arma::diagmat(pow_nb) * C_AO.t();
    arma::mat Kb = basis.exchange(Pb_pow);
    gC_xc += 4.0 * xc_prefactor * Kb * C_AO * arma::diagmat(pow_nb);
  }
  gC_out = gC_xc;
}

// XC occupation gradient
template <typename BasisType>
inline void muller_occupation_gradient(BasisType& basis,
                                      const arma::mat& C_AO,
                                      const arma::vec& n,
                                      double power,
                                      arma::vec& gn_out) {
  arma::uword Norb = C_AO.n_cols;
  arma::uword Nocc = n.n_elem;
  if (Norb == 0) { gn_out.reset(); return; }

  arma::vec na, nb;
  bool split_spin = false;
  if (Nocc == 2 * Norb) {
    split_spin = true;
    na = n.head(Norb);
    nb = n.tail(Norb);
  } else if (Nocc == Norb) {
    na = n;
    nb.zeros(Norb);
  } else {
    throw std::logic_error("muller_occupation_gradient: occupation vector size mismatch");
  }

  const double occ_eps = 1e-14;
  arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
  arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);

  arma::mat Pa = C_AO * arma::diagmat(na_eff) * C_AO.t();
  arma::mat Pb;
  if(split_spin) Pb = C_AO * arma::diagmat(nb_eff) * C_AO.t();
  arma::mat P_tot = split_spin ? (Pa + Pb) : Pa;

  arma::mat J = basis.coulomb(P_tot);

  arma::mat H_no = C_AO.t() * (basis.kinetic() + basis.nuclear()) * C_AO;
  arma::mat J_no = C_AO.t() * J * C_AO;

  arma::mat Pa_pow = C_AO * arma::diagmat(arma::pow(arma::clamp(na_eff, occ_eps, 1.0), power)) * C_AO.t();
  arma::mat Ka = basis.exchange(Pa_pow);
  arma::mat Ka_no = C_AO.t() * Ka * C_AO;

  arma::mat Kb_no;
  if(split_spin) {
    arma::mat Pb_pow = C_AO * arma::diagmat(arma::pow(arma::clamp(nb_eff, occ_eps, 1.0), power)) * C_AO.t();
    arma::mat Kb = basis.exchange(Pb_pow);
    Kb_no = C_AO.t() * Kb * C_AO;
  }

  gn_out.set_size(Nocc);
  auto compute_gn_spin = [&](arma::uword index, double n_val, const arma::mat& K_mat_no) -> double {
    double val = H_no(index, index) + J_no(index, index);
    double n_eff = std::max(occ_eps, std::min(1.0, n_val));
    val += 2.0 * (0.5) * power * std::pow(n_eff, power - 1.0) * K_mat_no(index, index);
    return val;
  };

  for(arma::uword i=0; i<Norb; ++i) {
    gn_out(i) = compute_gn_spin(i, na_eff(i), Ka_no);
  }
  if(split_spin) {
    for(arma::uword i=0; i<Norb; ++i) {
      gn_out(Norb + i) = compute_gn_spin(i, nb_eff(i), Kb_no);
    }
  }
}

} // namespace rdmft
} // namespace helfem

#endif // HELFEM_RDMFT_GRADIENTS_H

