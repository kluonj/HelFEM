#include "xc.h"
#include "../atomic/TwoDBasis.h" // Needed for explicit instantiation
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace helfem {
namespace rdmft {

XCFunctionalType string_to_xc_type(const std::string& type_str) {
    if (type_str == "Power") return XCFunctionalType::Power;
    if (type_str == "Muller") return XCFunctionalType::Muller;
    if (type_str == "HartreeFock" || type_str == "HF") return XCFunctionalType::HartreeFock;
    if (type_str == "GoedeckerUmrigar" || type_str == "GU") return XCFunctionalType::GoedeckerUmrigar;
    throw std::runtime_error("Unknown XC Functional Type: " + type_str);
}

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

template <typename BasisType>
double xc_energy(BasisType& basis,
                 const arma::mat& C_AO,
                 const arma::vec& n,
                 XCFunctionalType type,
                 double power,
                 int n_alpha) {
    
    // Recursive splitting for Unrestricted / Restricted-Spatial
    if(C_AO.n_cols == 0) return 0.0;
  
    // Unrestricted case with C containing [Ca, Cb]
    if (n_alpha > 0 && n_alpha < (int)C_AO.n_cols) {
      int Na = n_alpha;
      int Nb = C_AO.n_cols - Na;
      arma::mat Ca = C_AO.cols(0, Na - 1);
      arma::mat Cb = C_AO.cols(Na, Na + Nb - 1);
      arma::vec na = n.head(Na);
      arma::vec nb = n.tail(Nb);
      return xc_energy(basis, Ca, na, type, power, 0) + xc_energy(basis, Cb, nb, type, power, 0);
    }
  
    // Restricted Spatial with Spin-Resolved Occs (2*Norb)
    arma::uword Norb = C_AO.n_cols;
    if (n.n_elem == 2 * Norb) {
        arma::vec na = n.head(Norb);
        arma::vec nb = n.tail(Norb);
        return xc_energy(basis, C_AO, na, type, power, 0) + xc_energy(basis, C_AO, nb, type, power, 0);
    }
    
    if(n.n_elem != Norb) {
        throw std::logic_error("xc_energy: occupation vector size mismatch");
    }

    // Determine effective power
    double p = power;
    if (type == XCFunctionalType::Muller || type == XCFunctionalType::GoedeckerUmrigar) {
        p = 0.5;
    } else if (type == XCFunctionalType::HartreeFock) {
        p = 1.0;
    } 

    const double occ_eps = 1e-14;
    arma::vec n_eff = arma::clamp(n, 0.0, 1.0);
    arma::vec pow_n = arma::pow(arma::clamp(n_eff, occ_eps, 1.0), p);

    arma::mat P_pow = C_AO * arma::diagmat(pow_n) * C_AO.t();
    arma::mat K = basis.exchange(P_pow);
    double xc_val = 0.5 * arma::trace(P_pow * K); // Mullers term

    // Goedecker-Umrigar Correction
    if (type == XCFunctionalType::GoedeckerUmrigar) {
        double si_corr = 0.0;
        for (arma::uword i = 0; i < Norb; ++i) {
             arma::vec ci = C_AO.col(i);
             arma::mat Pi = ci * ci.t();
             // Self-interaction (ii|ii) = Tr(Pi * J[Pi])
             arma::mat J_Pi = basis.coulomb(Pi); // J[Pi]
             double Jii = arma::trace(Pi * J_Pi); 
             
             double ni = n_eff(i);
             // Correction: + 0.5 * (ni - ni^2) * Jii
             si_corr += 0.5 * (ni - ni * ni) * Jii;
        }
        xc_val += si_corr;
    }

    return xc_val;
}

template <typename BasisType>
void xc_orbital_gradient(BasisType& basis,
                         const arma::mat& C_AO,
                         const arma::vec& n,
                         XCFunctionalType type,
                         double power,
                         arma::mat& gC_out,
                         int n_alpha) {
    if(C_AO.n_cols == 0) { gC_out.reset(); return; }

    // Split spin
    if (n_alpha > 0) {
        int n_beta = C_AO.n_cols - n_alpha;
        arma::mat Ca = C_AO.head_cols(n_alpha);
        arma::mat Cb = C_AO.tail_cols(n_beta);
        arma::vec na = n.head(n_alpha);
        arma::vec nb = n.tail(n_beta);

        arma::mat gCa, gCb;
        xc_orbital_gradient(basis, Ca, na, type, power, gCa, 0);
        xc_orbital_gradient(basis, Cb, nb, type, power, gCb, 0);
        gC_out = arma::join_horiz(gCa, gCb);
        return;
    }

    arma::uword Norb = C_AO.n_cols;
    arma::vec na, nb;
    bool split_spin = false;
    if (n.n_elem == 2 * Norb) {
        split_spin = true;
        na = n.head(Norb);
        nb = n.tail(Norb);
    } else {
        na = n;
    }

    double p = power;
    if (type == XCFunctionalType::Muller || type == XCFunctionalType::GoedeckerUmrigar) p = 0.5;
    if (type == XCFunctionalType::HartreeFock) p = 1.0;

    const double occ_eps = 1e-14;
    arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
    arma::vec pow_na = arma::pow(arma::clamp(na_eff, occ_eps, 1.0), p);
    
    // Muller/Power Gradient
    // d/dC (0.5 Tr(P^p K[P^p])) = 2.0 * 0.5 * K[P^p] * C * diag(n^p) * 2 (Wait)
    // Actually, d(Tr(P K))/dP = 2K. dP/dC = C n + ...
    // gC = 4 * 0.5 * K * C * n^p = 2 * K * C * n^p
    arma::mat Pa_pow = C_AO * arma::diagmat(pow_na) * C_AO.t();
    arma::mat Ka = basis.exchange(Pa_pow);
    arma::mat gC_xc = 2.0 * Ka * C_AO * arma::diagmat(pow_na);

    if(split_spin) {
        arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);
        arma::vec pow_nb = arma::pow(arma::clamp(nb_eff, occ_eps, 1.0), p);
        arma::mat Pb_pow = C_AO * arma::diagmat(pow_nb) * C_AO.t();
        arma::mat Kb = basis.exchange(Pb_pow);
        gC_xc += 2.0 * Kb * C_AO * arma::diagmat(pow_nb);
    }

    // GU Correction Gradient
    if (type == XCFunctionalType::GoedeckerUmrigar) {
        // sum_i 0.5 (ni - ni^2) Jii
        // Jii = (ii|ii) = <i|J[Pi]|i>. 
        // dJii/dCi = ?
        // Jii = sum_ab sum_cd (Ci_a Ci_b) (ab|cd) (Ci_c Ci_d)
        // d/dCi = 4 * sum_bcd Ci_b (ab|cd) Ci_c Ci_d = 4 * [sum_d (sum_bc Ci_b (ab|cd) Ci_c) Ci_d]
        // = 4 * [J[Pi] * Ci]_a
        // So gC_i += 0.5 * (ni - ni^2) * 4 * J[Pi] * Ci
        //          = 2 * (ni - ni^2) * J[Pi] * Ci
        
        for (arma::uword i = 0; i < Norb; ++i) {
            arma::vec ci = C_AO.col(i);
            arma::mat Pi = ci * ci.t();
            arma::mat J_Pi = basis.coulomb(Pi);
            
            double ni = na_eff(i);
            double fac = 2.0 * 0.5 * (ni - ni * ni); // 2 from gradient, 0.5 from energy prefactor
            
            // Add to column i of gC_xc
            gC_xc.col(i) += fac * 2.0 * J_Pi * ci;
            
            if (split_spin) {
                 double val = nb(i);
                 double nib = (val < 0.0) ? 0.0 : ((val > 1.0) ? 1.0 : val);
                 double fac_b = (nib - nib * nib); 
                 // Same spatial orbital, different occupation factor
                 gC_xc.col(i) += fac_b * 2.0 * J_Pi * ci;
            }
        }
    }

    gC_out = gC_xc;
}

template <typename BasisType>
void xc_occupation_gradient(BasisType& basis,
                            const arma::mat& C_AO,
                            const arma::vec& n,
                            XCFunctionalType type,
                            double power,
                            arma::vec& gn_out,
                            int n_alpha) {
    if(C_AO.n_cols == 0) { gn_out.reset(); return; }

    // Split spin logic
    if (n_alpha > 0) {
        int n_beta = C_AO.n_cols - n_alpha;
        arma::mat Ca = C_AO.head_cols(n_alpha);
        arma::mat Cb = C_AO.tail_cols(n_beta);
        arma::vec na = n.head(n_alpha);
        arma::vec nb = n.tail(n_beta);
        
        arma::vec gna, gnb;
        xc_occupation_gradient(basis, Ca, na, type, power, gna, 0);
        xc_occupation_gradient(basis, Cb, nb, type, power, gnb, 0);
        gn_out = arma::join_vert(gna, gnb);
        return;
    }

    arma::uword Norb = C_AO.n_cols;
    arma::vec na, nb;
    bool split_spin = false;
    if (n.n_elem == 2 * Norb) {
        split_spin = true;
        na = n.head(Norb);
        nb = n.tail(Norb);
    } else {
        na = n;
    }

    double p = power;
    if (type == XCFunctionalType::Muller || type == XCFunctionalType::GoedeckerUmrigar) p = 0.5;
    if (type == XCFunctionalType::HartreeFock) p = 1.0;

    const double occ_eps = 1e-14;
    arma::vec na_eff = arma::clamp(na, 0.0, 1.0);
    // Muller Term: dE/dn_i = 0.5 * p * n_i^(p-1) * K_ii[P^p] ?
    // Energy E = 0.5 * sum_kl n_k^p n_l^p K_kl (in MO basis)
    // dE/dn_i = 0.5 * 2 * p * n_i^(p-1) * sum_l n_l^p K_il
    //         = p * n_i^(p-1) * (K[P^p])_ii
    
    // Need K matrix in MO basis
    arma::vec pow_na = arma::pow(arma::clamp(na_eff, occ_eps, 1.0), p);
    arma::mat Pa_pow = C_AO * arma::diagmat(pow_na) * C_AO.t();
    arma::mat Ka = basis.exchange(Pa_pow);
    // Project to MO: C^T K C
    arma::mat Ka_no = C_AO.t() * Ka * C_AO;

    arma::vec gn_xc(n.n_elem);
    
    for(arma::uword i=0; i<Norb; ++i) {
        double ni = na_eff(i);
        double term = 0.0;
        if (ni > occ_eps) {
            term = p * std::pow(ni, p - 1.0) * Ka_no(i,i);
        }
        gn_xc(i) = term;
    }

    if(split_spin) {
        arma::vec nb_eff = arma::clamp(nb, 0.0, 1.0);
        arma::vec pow_nb = arma::pow(arma::clamp(nb_eff, occ_eps, 1.0), p);
        arma::mat Pb_pow = C_AO * arma::diagmat(pow_nb) * C_AO.t();
        arma::mat Kb = basis.exchange(Pb_pow);
        arma::mat Kb_no = C_AO.t() * Kb * C_AO;
        
        for(arma::uword i=0; i<Norb; ++i) {
            double ni = nb_eff(i);
            double term = 0.0;
            if (ni > occ_eps) {
                term = p * std::pow(ni, p - 1.0) * Kb_no(i,i);
            }
            gn_xc(Norb + i) = term;
        }
    }

    // GU Correction
    if (type == XCFunctionalType::GoedeckerUmrigar) {
        // E += 0.5 * (ni - ni^2) * Jii
        // dE/dni = 0.5 * (1 - 2*ni) * Jii
        for (arma::uword i = 0; i < Norb; ++i) {
             arma::vec ci = C_AO.col(i);
             arma::mat Pi = ci * ci.t();
             arma::mat J_Pi = basis.coulomb(Pi);
             double Jii = arma::trace(Pi * J_Pi); 
             
             double ni = na_eff(i);
             gn_xc(i) += 0.5 * (1.0 - 2.0 * ni) * Jii;

             if (split_spin) {
                 double val = nb(i);
                 double nib = (val < 0.0) ? 0.0 : ((val > 1.0) ? 1.0 : val);
                 gn_xc(Norb + i) += 0.5 * (1.0 - 2.0 * nib) * Jii;
             }
        }
    }
    
    gn_out = gn_xc;
}


// Explicit instantiations
template double xc_energy<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, XCFunctionalType, double, int);
template void xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, XCFunctionalType, double, arma::mat&, int);
template void xc_occupation_gradient<helfem::atomic::basis::TwoDBasis>(helfem::atomic::basis::TwoDBasis&, const arma::mat&, const arma::vec&, XCFunctionalType, double, arma::vec&, int);

} // namespace rdmft
} // namespace helfem
