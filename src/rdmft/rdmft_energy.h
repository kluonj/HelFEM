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
double core_energy(const arma::mat& Hcore,
                   const arma::mat& C_AO,
                   const arma::vec& n);

// Hartree (Coulomb) energy: E_J = 0.5 * trace(P * J)
template <typename BasisType>
double hartree_energy(BasisType& basis,
                      const arma::mat& C_AO,
                      const arma::vec& n);

// Müller/power XC energy helper
template <typename BasisType>
double xc_energy(BasisType& basis,
                 const arma::mat& C_AO,
                 const arma::vec& n,
                 double power);

// Compute total energy and all gradients
// This matches the signature of EnergyFunctional::energy but takes Hcore/Basis/power explicitly.
template <typename BasisType>
double compute_energy_and_gradients(BasisType& basis,
                                    const arma::mat& Hcore,
                                    const arma::mat& C_AO,
                                    const arma::vec& n,
                                    double power,
                                    arma::mat& gC,
                                    arma::vec& gn);

} // namespace rdmft
} // namespace helfem

#endif // HELFEM_RDMFT_ENERGY_H
