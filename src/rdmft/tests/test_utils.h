#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <armadillo>
#include "general/constants.h"
#include "general/checkpoint.h"
#include "atomic/basis.h"
#include "atomic/dftgrid.h"
#include "general/model_potential.h"
#include "general/scf_helpers.h"
#include "rdmft/solver.h"
#include "rdmft/energy.h"
#include "rdmft/gradients.h"

using namespace helfem;
using namespace helfem::rdmft;

// RDMFT Functional for testing (Unrestricted)
class TestRDMFTFunctional : public EnergyFunctional<void> {
public:
    helfem::atomic::basis::TwoDBasis& basis;
    arma::mat Hcore;
    double accumulated_energy;
    double power;
    int n_alpha_orb;
    XCFunctionalType xc_type;

    TestRDMFTFunctional(helfem::atomic::basis::TwoDBasis& b, const arma::mat& H0, int na_orb, double p=1.0, XCFunctionalType type = XCFunctionalType::Power) 
        : basis(b), Hcore(H0), accumulated_energy(0.0), power(p), n_alpha_orb(na_orb), xc_type(type) {}

    double energy(const arma::mat& C, const arma::vec& n) override {
        // Use unified helper functions which handle n_alpha_orb splitting and cross-terms
        accumulated_energy = helfem::rdmft::compute_energy<helfem::atomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, n_alpha_orb, xc_type);
        return accumulated_energy;
    }

    void orbital_gradient(const arma::mat& C, const arma::vec& n, arma::mat& gC) override {
        helfem::rdmft::compute_orbital_gradient<helfem::atomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, gC, n_alpha_orb, xc_type);
    }

    void occupation_gradient(const arma::mat& C, const arma::vec& n, arma::vec& gn) override {
        helfem::rdmft::compute_occupation_gradient<helfem::atomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, gn, n_alpha_orb, xc_type);
    }

    // Backwards-compatible overload: compute energy and optionally gradients (old API)
    double energy(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn) {
        double E = energy(C, n);
        orbital_gradient(C, n, gC);
        occupation_gradient(C, n, gn);
        return E;
    }
};

// Compute HF energy for given orbitals and occupations
// NOTE: This mirrors the energy expression used in perform_scf for HF.
inline double compute_hf_energy(helfem::atomic::basis::TwoDBasis& basis,
                                const arma::mat& H0,
                                const arma::mat& Ca,
                                const arma::mat& Cb,
                                const arma::vec& na,
                                const arma::vec& nb) {
    arma::mat Pa = Ca * arma::diagmat(na) * Ca.t();
    arma::mat Pb = Cb * arma::diagmat(nb) * Cb.t();
    arma::mat Ptot = Pa + Pb;

    arma::mat J = basis.coulomb(Ptot);
    arma::mat Ka = basis.exchange(Pa);
    arma::mat Kb = basis.exchange(Pb);

    double E_core = arma::trace(Ptot * H0);
    double E_J = 0.5 * arma::trace(Ptot * J);
    double E_Exx = 0.5 * (arma::trace(Pa * Ka) + arma::trace(Pb * Kb));

    return E_core + E_J + E_Exx;
}

// Helper to run SCF (HF or DFT) to get converged orbitals
// Returns pair {Ca, Cb}
/* 
NOTE: The implementation of this is identical to previous test_solver.cpp.
Keeping it inline here for simplicity. 
*/
inline std::pair<arma::mat, arma::mat> perform_scf(helfem::atomic::basis::TwoDBasis& basis, 
                                                const arma::mat& S, 
                                                const arma::mat& H0, 
                                                int Na, int Nb,
                                                int x_func = 0, int c_func = 0) {
    bool do_dft = (x_func > 0 || c_func > 0);
    std::string method = do_dft ? "DFT" : "HF";
    std::cout << "Starting " << method << " SCF Pre-optimization...\n";
    
    arma::mat Sinvh = helfem::scf::form_Sinvh(S, false);
    
    // Guess using Thomas-Fermi atom (like atomic/main.cpp iguess==3)
    arma::vec eps; arma::mat C;
    modelpotential::ModelPotential * model = new modelpotential::TFAtom(basis.get_Z());
    arma::mat Vel = arma::mat(basis.Nbf(), basis.Nbf(), arma::fill::zeros);
    arma::mat Vmag = arma::mat(basis.Nbf(), basis.Nbf(), arma::fill::zeros);
    arma::mat Hguess = basis.kinetic() + Vel + Vmag + basis.model_potential(model);
    delete model;
    helfem::scf::eig_gsym(eps, C, Hguess, Sinvh);
    arma::mat Ca = C;
    arma::mat Cb = C;
    
    // Break symmetry slightly for open shell if Na != Nb or to ensure UHF solution
    if (Na != Nb) {
        // Already broken symmetry in occupation
    } else {
        // Perturb Beta slightly
         Cb.col(0) = 0.9*Cb.col(0) + 0.1*Cb.col(Cb.n_cols-1);
    }
    
    int maxit = 50;
    double tol = 1e-6;
    double E_old = 0.0;
    
    // Initialize DFT Grid if needed
    helfem::atomic::dftgrid::DFTGrid grid;
    arma::vec x_pars, c_pars;
    
    if (do_dft) {
        int lmax = arma::max(basis.get_lval());
        int mmax = arma::max(basis.get_mval());
        int ldft = 4*lmax + 10;
        int mdft = 4*mmax + 5;
        grid = helfem::atomic::dftgrid::DFTGrid(&basis, ldft, mdft);
    }
    
    arma::mat Pa_old, Pb_old;
    
    for(int iter=1; iter<=maxit; ++iter) {
        arma::mat Ca_occ = Ca.cols(0, Na-1);
        arma::mat Cb_occ = (Nb > 0) ? Cb.cols(0, Nb-1) : arma::mat();
        
        arma::mat Pa = Ca_occ * Ca_occ.t();
        arma::mat Pb = (Nb > 0) ? Cb_occ * Cb_occ.t() : arma::mat(Pa.n_rows, Pa.n_cols, arma::fill::zeros);
        
        // Simple Density Mixing
        if (iter > 1) {
            double alpha = 0.5;
            Pa = (1.0 - alpha) * Pa_old + alpha * Pa;
            Pb = (1.0 - alpha) * Pb_old + alpha * Pb;
        }
        Pa_old = Pa;
        Pb_old = Pb;
        
        arma::mat Ptot = Pa + Pb;
        arma::mat J = basis.coulomb(Ptot);
        
        // Exact Exchange (HF only)
        arma::mat Ka(J.n_rows, J.n_cols, arma::fill::zeros);
        arma::mat Kb(J.n_rows, J.n_cols, arma::fill::zeros);
        double E_Exx = 0.0;
        
        if (!do_dft) {
            Ka = basis.exchange(Pa);
            Kb = (Nb > 0) ? basis.exchange(Pb) : arma::mat(J.n_rows, J.n_cols, arma::fill::zeros);
            E_Exx = 0.5 * (arma::trace(Pa * Ka) + arma::trace(Pb * Kb));
        }

        // DFT Exchange-Correlation
        double E_Exc = 0.0;
        arma::mat XCa(J.n_rows, J.n_cols, arma::fill::zeros);
        arma::mat XCb(J.n_rows, J.n_cols, arma::fill::zeros);

        if (do_dft) {
            double Nel = 0.0;
            double Ekin_dft = 0.0;
            double dftthr = 1e-12;
            
            // eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb, Ha, Hb, Exc, Nel, Ekin, beta, thr)
            grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pa, Pb, XCa, XCb, E_Exc, Nel, Ekin_dft, (Nb>0), dftthr);
        }
        
        // Fock Matrix construction
        arma::mat Fa = H0 + J + Ka + XCa;
        arma::mat Fb = H0 + J + Kb + XCb;
        
        double E_kin = arma::trace(Ptot * basis.kinetic());
        double E_nuc = arma::trace(Ptot * basis.nuclear());
        double E_J   = 0.5 * arma::trace(Ptot * J);
        double E_tot = E_kin + E_nuc + E_J + E_Exx + E_Exc;
        
        if (std::abs(E_tot - E_old) < tol) {
            std::cout << method << " Converged at Iter " << iter << " E=" << E_tot << "\n";
            return {Ca, Cb};
        }
        E_old = E_tot;
        if(iter==1 || iter%1==0) std::cout << "  Iter " << iter << " E=" << E_tot << "\n";
        
        helfem::scf::eig_gsym(eps, Ca, Fa, Sinvh);
        if (Nb > 0) helfem::scf::eig_gsym(eps, Cb, Fb, Sinvh);
        else Cb = Ca; 
    }
    
    std::cout << method << " SCF did not converge fully.\n";
    return {Ca, Cb};
}

#endif // TEST_UTILS_H
