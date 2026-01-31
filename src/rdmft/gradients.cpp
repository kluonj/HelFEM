#include "gradients.h"
#include "xc.h"
#include "../atomic/TwoDBasis.h"
#include "../diatomic/basis.h"
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
  for (int it = 0; it < 20; ++it) {
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
void compute_orbital_gradient(BasisType& basis,
                              const arma::mat& H_core,
                              const arma::mat& C_AO,
                              const arma::vec& n,
                              double power,
                              arma::mat& gC_out,
                              int n_alpha,
                              XCFunctionalType xc_type) {
    arma::mat gC_core;
    core_orbital_gradient(H_core, C_AO, n, gC_core);
    
    arma::mat gC_hartree;
    hartree_orbital_gradient(basis, C_AO, n, gC_hartree);
    
    arma::mat gC_xc;
    xc_orbital_gradient(basis, C_AO, n, xc_type, power, gC_xc, n_alpha);
    
    gC_out = gC_core + gC_hartree + gC_xc;
}

template <typename BasisType>
void compute_occupation_gradient(BasisType& basis,
                                 const arma::mat& H_core,
                                 const arma::mat& C_AO,
                                 const arma::vec& n,
                                 double power,
                                 arma::vec& gn_out,
                                 int n_alpha,
                                 XCFunctionalType xc_type) {
    arma::vec gn_core;
    core_occupation_gradient(H_core, C_AO, n, gn_core);
    
    arma::vec gn_hartree;
    hartree_occupation_gradient(basis, C_AO, n, gn_hartree);
    
    arma::vec gn_xc;
    xc_occupation_gradient(basis, C_AO, n, xc_type, power, gn_xc, n_alpha);
    
    gn_out = gn_core + gn_hartree + gn_xc;
}

// Explicit instantiations
template void hartree_orbital_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::mat&);
template void hartree_occupation_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::vec&);

// xc_... explicit instantiations moved to xc.cpp

template void compute_orbital_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, arma::mat&, int, XCFunctionalType);
template void compute_occupation_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, arma::vec&, int, XCFunctionalType);

// Diatomic instantiations
template void hartree_orbital_gradient<helfem::diatomic::basis::TwoDBasis>(helfem::diatomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::mat&);
template void hartree_occupation_gradient<helfem::diatomic::basis::TwoDBasis>(helfem::diatomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, arma::vec&);

template void compute_orbital_gradient<helfem::diatomic::basis::TwoDBasis>(helfem::diatomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, arma::mat&, int, XCFunctionalType);
template void compute_occupation_gradient<helfem::diatomic::basis::TwoDBasis>(helfem::diatomic::basis::TwoDBasis&, const arma::mat&, const arma::mat&, const arma::vec&, double, arma::vec&, int, XCFunctionalType);

} // namespace rdmft
} // namespace helfem
