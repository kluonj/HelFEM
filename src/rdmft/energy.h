// energy.h
// Defines a minimal, reusable interface for RDMFT energy functionals.

#ifndef ENERGY_H
#define ENERGY_H

#include <armadillo>
#include "xc.h" // Include the XC definition

namespace helfem {
namespace rdmft {

// Minimal abstract interface for an energy functional usable by external
// code: computes energy and returns Euclidean gradients w.r.t. orbitals and
// occupations.
template <typename BasisType>
struct EnergyFunctional {
  // Evaluate energy: given AO coeffs `C` (columns = orbitals) and occupation
  // vector `n`, return energy.
  virtual double energy(const arma::mat& C, const arma::vec& n) = 0;

  // Evaluate orbital gradient w.r.t C
  virtual void orbital_gradient(const arma::mat& C, const arma::vec& n, arma::mat& gC) = 0;

  // Evaluate occupation gradient w.r.t n
  virtual void occupation_gradient(const arma::mat& C, const arma::vec& n, arma::vec& gn) = 0;

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

// XC energy helper
// (Defined in xc.h)


// Compute total energy: E_core + E_Hartree + E_XC
template <typename BasisType>
double compute_energy(BasisType& basis,
                      const arma::mat& Hcore,
                      const arma::mat& C_AO,
                      const arma::vec& n,
                      double power,
                      int n_alpha = 0,
                      XCFunctionalType type = XCFunctionalType::Power);

} // namespace rdmft
} // namespace helfem

#endif // ENERGY_H
