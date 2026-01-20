#include <iostream>
#include <armadillo>
#include "general/constants.h"
#include "general/checkpoint.h"
#include "atomic/basis.h"
#include "atomic/dftgrid.h"
#include "general/model_potential.h"
#include "general/scf_helpers.h"
#include "rdmft/rdmft_solver.h"
#include "rdmft/rdmft_energy.h"
#include "rdmft/rdmft_gradients.h"

using namespace std;
using namespace helfem;
using namespace helfem::rdmft;

// Simplified test: only UnrestrictedAtomicFunctional and UHF workflow retained

// Unrestricted Atomic Functional
class UnrestrictedAtomicFunctional : public EnergyFunctional<void> {
public:
    helfem::atomic::basis::TwoDBasis& basis;
    arma::mat Hcore;
    double accumulated_energy;
    int n_alpha_orb;

    UnrestrictedAtomicFunctional(helfem::atomic::basis::TwoDBasis& b, const arma::mat& H0, int na_orb) 
        : basis(b), Hcore(H0), accumulated_energy(0.0), n_alpha_orb(na_orb) {}

    double energy(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn) override {
        double power = 1.0;
        
        // Split C and n
        // C has (n_alpha_orb + n_beta_orb) columns.
        // We assume n corresponds similarly.
        
        int Na = n_alpha_orb;
        int Nb = C.n_cols - Na;
        
        if (Nb < 0) throw std::logic_error("UnrestrictedAtomicFunctional: invalid n_alpha_orb");
        
        arma::mat Ca = C.cols(0, Na - 1);
        arma::mat Cb = C.cols(Na, Na + Nb - 1);
        
        arma::vec na = n.head(Na);
        arma::vec nb = n.tail(Nb);
        
        // Form Density Matrices
        arma::mat Pa = Ca * arma::diagmat(na) * Ca.t();
        arma::mat Pb = Cb * arma::diagmat(nb) * Cb.t();
        arma::mat Ptot = Pa + Pb;
        
        // Use helper functions for energy components and gradients (reuse implementations)
        // Compute per-channel energies via helpers and sum.
        double E_core_a = helfem::rdmft::core_energy(Hcore, Ca, na);
        double E_core_b = helfem::rdmft::core_energy(Hcore, Cb, nb);
        double E_core = E_core_a + E_core_b;

        // Hartree must be computed from total density Ptot to include cross terms
        arma::mat J_total = basis.coulomb(Ptot);
        double E_J = 0.5 * arma::trace(Ptot * J_total);

        double E_xc_a = helfem::rdmft::xc_energy<helfem::atomic::basis::TwoDBasis>(basis, Ca, na, power);
        double E_xc_b = helfem::rdmft::xc_energy<helfem::atomic::basis::TwoDBasis>(basis, Cb, nb, power);
        double E_xc = E_xc_a + E_xc_b;

        accumulated_energy = E_core + E_J + E_xc;

        // Orbital gradients via helpers (they operate on per-channel C and occupations)
        arma::mat gCa_core, gCa_J, gCa_xc;
        arma::mat gCb_core, gCb_J, gCb_xc;

        helfem::rdmft::core_orbital_gradient(Hcore, Ca, na, gCa_core);
        helfem::rdmft::core_orbital_gradient(Hcore, Cb, nb, gCb_core);

        // Hartree orbital gradients computed from total coulomb potential J_total
        gCa_J = 2.0 * J_total * Ca * arma::diagmat(na);
        gCb_J = 2.0 * J_total * Cb * arma::diagmat(nb);

        helfem::rdmft::muller_xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, Ca, na, power, gCa_xc);
        helfem::rdmft::muller_xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, Cb, nb, power, gCb_xc);

        gC.set_size(arma::size(C));
        gC.cols(0, Na-1) = gCa_core + gCa_J + gCa_xc;
        gC.cols(Na, Na+Nb-1) = gCb_core + gCb_J + gCb_xc;

        // Occupation gradients via helper (per-channel)
        arma::vec gna_full, gnb_full;
        helfem::rdmft::muller_occupation_gradient<helfem::atomic::basis::TwoDBasis>(basis, Ca, na, power, gna_full);
        helfem::rdmft::muller_occupation_gradient<helfem::atomic::basis::TwoDBasis>(basis, Cb, nb, power, gnb_full);

        // The helper outputs H + J(P_channel) + dExc/dn for that channel where J(P_channel)
        // is constructed from Pa (or Pb). We need H + J_total + dExc/dn where J_total is built
        // from Ptot = Pa + Pb. So correct by adding (J_total_diag - J_channel_diag).
        arma::mat J_pa = basis.coulomb(Pa);
        arma::mat J_pb = basis.coulomb(Pb);
        arma::mat J_pa_no = Ca.t() * J_pa * Ca;
        arma::mat J_pb_no = Cb.t() * J_pb * Cb;
        arma::mat J_tot_a_no = Ca.t() * J_total * Ca;
        arma::mat J_tot_b_no = Cb.t() * J_total * Cb;

        arma::vec gn_a = gna_full + (J_tot_a_no.diag() - J_pa_no.diag());
        arma::vec gn_b = gnb_full + (J_tot_b_no.diag() - J_pb_no.diag());

        gn.set_size(Na + Nb);
        gn.head(Na) = gn_a;
        gn.tail(Nb) = gn_b;
        
        return accumulated_energy;
    }
};

// Helper to run SCF (HF or DFT) to get converged orbitals
// Returns pair {Ca, Cb}
std::pair<arma::mat, arma::mat> perform_dft_scf(helfem::atomic::basis::TwoDBasis& basis, 
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

void run_test_unrestricted(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Unrestricted Calculation (DFT Orbitals, Uniform Occ Optimization) ---\n";
    
    // 1. Run DFT (VWN) SCF to get guess orbitals for He (Na=1, Nb=1)
    int Na = 1; int Nb = 1;
    
    // Using x_func=1 (Slater Exchange) and c_func=7 (VWN Correlation LDA)
    // LibXC IDs: XC_LDA_X=1, XC_LDA_C_VWN=7
    int x_func_id = 1; 
    int c_func_id = 7;
    
    auto orbits = perform_dft_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca_hf = orbits.first;
    arma::mat Cb_hf = orbits.second;
    
    int Na_orb = Ca_hf.n_cols; 
    int Nb_orb = Cb_hf.n_cols;
    
    auto func = std::make_shared<UnrestrictedAtomicFunctional>(basis, H0, Na_orb);
    RDMFT_Solver solver(func, S);
    
    // Guess: DFT Orbitals
    arma::mat C_tot(Ca_hf.n_rows, Na_orb + Nb_orb);
    C_tot.cols(0, Na_orb - 1) = Ca_hf;
    C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb_hf;
    
    // Uniform occupations (fractional)
    double target_Na = (double)Na;
    double target_Nb = (double)Nb;
    
    arma::vec na(Na_orb); na.fill(target_Na / double(Na_orb));
    arma::vec nb(Nb_orb); nb.fill(target_Nb / double(Nb_orb));
    
    arma::vec n_tot(Na_orb + Nb_orb);
    n_tot.head(Na_orb) = na;
    n_tot.tail(Nb_orb) = nb;
    
    std::cout << "Initial Occupations (Uniform): " << n_tot.head(5).t();
    
    solver.set_verbose(true);
    solver.set_optimize_orbitals(false); // FIXED ORBITALS
    
    // Solve with dual channel
    solver.solve(C_tot, n_tot, target_Na, target_Nb, Na_orb);
    
    std::cout << "Unrestricted Final Energy: " << func->accumulated_energy << "\n";
    
    // Print first few occupations
    int n_print = std::min(5, Na_orb);
    std::cout << "Occupations Alpha (first " << n_print << "): " << n_tot.subvec(0, n_print-1).t();
    std::cout << "Occupations Beta  (first " << n_print << "): " << n_tot.subvec(Na_orb, Na_orb+n_print-1).t();
}

int main() {
    std::cout << "Testing RDMFT Solver Suite\n";
    
    // Setup Basis
    int Z = 2; // He
    int primbas = 4; int Nnodes = 15; int Nelem = 10; double Rmax = 20.0;
    
    auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
    arma::vec bval = helfem::atomic::basis::form_grid((helfem::modelpotential::nuclear_model_t)0, 0.0, Nelem, Rmax, 
                                                      4, 2.0, Nelem, 4, 2.0, 
                                                      2, 0, 0, 0.0, false, 0.0);
    
    arma::ivec lvals(1); lvals(0) = 0;
    arma::ivec mvals(1); mvals(0) = 0;
    
    helfem::atomic::basis::TwoDBasis basis(Z, (helfem::modelpotential::nuclear_model_t)0, 0.0, 
                                           poly, false, 5*poly->get_nbf(), bval, 
                                           poly->get_nprim()-1, lvals, mvals, 0, 0, 0.0);
                                           
    basis.compute_tei(true);
    arma::mat S = basis.overlap();
    arma::mat T = basis.kinetic();
    arma::mat Vnuc = basis.nuclear();
    arma::mat H0 = T + Vnuc;

    // run_test_restricted(basis, S, H0); // Disable for now to focus on UHF workflow
    run_test_unrestricted(basis, S, H0);
    
    return 0;
}
