#include <algorithm>
#include <string>
#include <iomanip>
#include "test_utils.h"

// Optimize occupations only (fixed orbitals)
// Then check if energy matches a reference HF calculation.
void optimize_occ_only(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Unrestricted Calculation (Occupation Optimization Only, p=1.0) ---\n";
    
    int Na = 1; int Nb = 1;
    double target_Na = (double)Na;
    double target_Nb = (double)Nb;

    // Integer occupations expected from DFT SCF for closed-shell He
    auto make_integer_occ = [](int n_orb, int n_elec) {
        arma::vec n(n_orb, arma::fill::zeros);
        for (int i = 0; i < n_elec && i < n_orb; ++i) n(i) = 1.0;
        return n;
    };

    // 1. Run HF SCF to get fixed orbitals for occ-only optimization
    std::cout << "\n>>> Running HF SCF to get fixed orbitals...\n";
    int x_func_id = 0; 
    int c_func_id = 0;
    auto orbits_hf = perform_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca_hf = orbits_hf.first;
    arma::mat Cb_hf = orbits_hf.second;

    auto run_occ_optimization = [&](const arma::mat& Ca, const arma::mat& Cb, const std::string& label) {
        int Na_orb = Ca.n_cols;
        int Nb_orb = Cb.n_cols;

        double power = 1.0; // RDMFT with p=0.5 is Muller
        auto func = std::make_shared<TestRDMFTFunctional>(basis, H0, Na_orb, power);
        Solver solver(func, S);

        arma::mat C_tot(Ca.n_rows, Na_orb + Nb_orb);
        C_tot.cols(0, Na_orb - 1) = Ca;
        C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb;

        arma::vec na0(Na_orb); na0.fill(target_Na / double(Na_orb));
        arma::vec nb0(Nb_orb); nb0.fill(target_Nb / double(Nb_orb));

        arma::vec n_tot(Na_orb + Nb_orb);
        n_tot.head(Na_orb) = na0;
        n_tot.tail(Nb_orb) = nb0;

        std::cout << "\n[" << label << "] Initial Occupations (Uniform): " << n_tot.head(5).t();

        solver.set_verbose(true);
        solver.set_optimize_orbitals(false); // FIXED ORBITALS
        solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::CG);
        solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::Armijo);

        solver.solve(C_tot, n_tot, target_Na + target_Nb);

        // Print energy with more digits (temporary formatting)
        std::cout << std::setprecision(12) << std::scientific;
        std::cout << "[" << label << "] Final Energy (p=1.0): " << func->accumulated_energy << "\n";
        std::cout << std::defaultfloat << std::setprecision(6);

        int n_print = std::min(5, Na_orb);
        std::cout << "[" << label << "] Occupations Alpha (first " << n_print << "): " << n_tot.subvec(0, n_print-1).t();
        std::cout << "[" << label << "] Occupations Beta  (first " << n_print << "): " << n_tot.subvec(Na_orb, Na_orb+n_print-1).t();
    };

    // 2. Occupation optimization with HF orbitals
    run_occ_optimization(Ca_hf, Cb_hf, "HF orbitals");
}

int main() {
    std::cout << "Testing RDMFT Solver: Occupation Optimization Only\n";
    
    // Setup Basis
    int Z = 2; // He
    int primbas = 4; int Nnodes = 10; int Nelem = 6; double Rmax = 40.0;
    
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

    optimize_occ_only(basis, S, H0);
    
    return 0;
}
