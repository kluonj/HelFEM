#ifndef RDMFT_XC_H
#define RDMFT_XC_H

#include <armadillo>
#include <string>

namespace helfem {
namespace rdmft {

enum class XCFunctionalType {
    Power,             // Generic n^p. Needs 'power' parameter.
    Muller,            // Power with p=0.5
    HartreeFock,       // Power with p=1.0
    GoedeckerUmrigar   // Muller + Self-Interaction Correction
};

// Helper: Convert string to enum (Power, Muller, HF, GU)
XCFunctionalType string_to_xc_type(const std::string& type_str);

// XC Energy
template <typename BasisType>
double xc_energy(BasisType& basis,
                 const arma::mat& C_AO,
                 const arma::vec& n,
                 XCFunctionalType type,
                 double power = 0.5,
                 int n_alpha = 0);

// XC Orbital Gradient
template <typename BasisType>
void xc_orbital_gradient(BasisType& basis,
                         const arma::mat& C_AO,
                         const arma::vec& n,
                         XCFunctionalType type,
                         double power,
                         arma::mat& gC_out,
                         int n_alpha = 0);

// XC Occupation Gradient
template <typename BasisType>
void xc_occupation_gradient(BasisType& basis,
                            const arma::mat& C_AO,
                            const arma::vec& n,
                            XCFunctionalType type,
                            double power,
                            arma::vec& gn_out,
                            int n_alpha = 0);

} // namespace rdmft
} // namespace helfem

#endif // RDMFT_XC_H
