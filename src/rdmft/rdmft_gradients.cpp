#include "rdmft_gradients.h"
#include "../atomic/TwoDBasis.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace helfem {
namespace rdmft {

arma::vec occ_from_theta(const arma::vec& theta) {
  return arma::square(arma::cos(theta));
}

arma::vec d_occ_d_theta(const arma::vec& theta) {
  return -arma::sin(2.0 * theta);
}

// Project onto capped simplex 0<=x<=1, sum(x)=target_sum
arma::vec project_capped_simplex(const arma::vec& y, double target_sum, double* lambda_out) {
  if (!y.is_finite()) throw std::logic_error("project_capped_simplex: non-finite input");
  if (target_sum < 0.0) throw std::logic_error("project_capped_simplex: target_sum < 0");
  if (target_sum > double(y.n_elem)) throw std::logic_error("project_capped_simplex: target_sum too large");

  auto sum_clamped = [&](double lambda) {
    arma::vec x = y - lambda;
    x.transform([](double v) { return std::min(1.0, std::max(0.0, v)); });
    return arma::sum(x);
  };

  double lo = y.min() - 1.0;
  double hi = y.max();
  if (sum_clamped(lo) < target_sum) lo -= 10.0;
  if (sum_clamped(hi) > target_sum) hi += 10.0;
  for (int it = 0; it < 80; ++it) {
    double mid = 0.5 * (lo + hi);
    double s = sum_clamped(mid);
    if (s > target_sum) lo = mid; else hi = mid;
  }
  double lambda_val = 0.5 * (lo + hi);
  if (lambda_out) *lambda_out = lambda_val;
  
  arma::vec x = y - lambda_val;
  x.transform([](double v) { return std::min(1.0, std::max(0.0, v)); });

  double diff = arma::sum(x) - target_sum;
  if (std::abs(diff) > 1e-10) {
    arma::uvec free = arma::find((x > 1e-12) % (x < 1.0 - 1e-12));
    if (!free.empty()) {
      x(free) -= diff / double(free.n_elem);
      x.transform(
        [](double v) { return std::min(1.0, std::max(0.0, v)); }
    );
    }
  }
  return x;
}

void core_orbital_gradient(const arma::mat& Hcore,
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
    throw std::logic_error("core_orbital_gradient: occupation vector size mismatch");
  }
  gC_out = 2.0 * Hcore * C_AO * arma::diagmat(n_sum);
}

void core_occupation_gradient(const arma::mat& Hcore,
                                    const arma::mat& C_AO,
                                    const arma::vec& n,
                                    arma::vec& gn_out) {
  if(C_AO.n_cols == 0) { gn_out.reset(); return; }
  arma::uword Norb = C_AO.n_cols;
  arma::mat H_no = C_AO.t() * Hcore * C_AO;
  arma::uword Nocc = n.n_elem;
  if (Nocc == Norb) {
    gn_out.set_size(Norb);
    for(arma::uword i=0;i<Norb;++i) gn_out(i) = H_no(i,i);
  } else if (Nocc == 2 * Norb) {
    gn_out.set_size(2*Norb);
    for(arma::uword i=0;i<Norb;++i) {
      gn_out(i) = H_no(i,i);
      gn_out(Norb + i) = H_no(i,i);
    }
  } else {
    throw std::logic_error("core_occupation_gradient: occupation vector size mismatch");
  }
}

template <typename BasisType>
void hartree_orbital_gradient(BasisType& basis,
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

template <typename BasisType>
void hartree_occupation_gradient(BasisType& basis,
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

template <typename BasisType>
void xc_orbital_gradient(BasisType& basis,
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
    throw std::logic_error("xc_orbital_gradient: occupation vector size mismatch");
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

template <typename BasisType>
void xc_occupation_gradient(BasisType& basis,
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
    throw std::logic_error("xc_occupation_gradient: occupation vector size mismatch");
  }

  const double occ_eps = 1e-14;
  arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
  arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);

  arma::mat Pa = C_AO * arma::diagmat(na_eff) * C_AO.t();

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
    double n_eff = std::max(occ_eps, std::min(1.0, n_val));
    // XC Gradient: - alpha * n^(alpha-1) * K_tilde_kk
    // Note: K_mat_no is -K_tilde (because basis.exchange returns -K)
    // So term is: + alpha * n^(alpha-1) * K_mat_no
    double val = 2.0 * (0.5) * power * std::pow(n_eff, power - 1.0) * K_mat_no(index, index);
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

// Explicit instantiations
template void hartree_orbital_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::mat&);
template void hartree_occupation_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::vec&);
template void xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, double, arma::mat&);
template void xc_occupation_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, double, arma::vec&);

} // namespace rdmft
} // namespace helfem
