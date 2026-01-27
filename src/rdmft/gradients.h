// gradients.h
// Small helpers for gradients and occupation transformations used by optimizers.

#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <armadillo>
#include "xc.h"

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
void hartree_orbital_gradient(BasisType& basis,
                                     const arma::mat& C_AO,
                                     const arma::vec& n,
                                     arma::mat& gC_out);

// Hartree occupation gradient
template <typename BasisType>
void hartree_occupation_gradient(BasisType& basis,
                                       const arma::mat& C_AO,
                                       const arma::vec& n,
                                       arma::vec& gn_out);

// XC orbital gradient (Müller/power)
// (Defined in xc.h)

// XC occupation gradient
// (Defined in xc.h)

// Compute total orbital gradient: gC_core + gC_Hartree + gC_XC
template <typename BasisType>
void compute_orbital_gradient(BasisType& basis,
                              const arma::mat& Hcore,
                              const arma::mat& C_AO,
                              const arma::vec& n,
                              double power,
                              arma::mat& gC_out,
                              int n_alpha = 0,
                              XCFunctionalType type = XCFunctionalType::Power);

// Compute total occupation gradient: gn_core + gn_Hartree + gn_XC
template <typename BasisType>
void compute_occupation_gradient(BasisType& basis,
                                 const arma::mat& Hcore,
                                 const arma::mat& C_AO,
                                 const arma::vec& n,
                                 double power,
                                 arma::vec& gn_out,
                                 int n_alpha = 0,
                                 XCFunctionalType type = XCFunctionalType::Power);

} // namespace rdmft
} // namespace helfem

#endif // GRADIENTS_H

