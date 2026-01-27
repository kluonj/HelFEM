#include <algorithm>
#include <string>
#include <iomanip>
#include "test_utils.h"

// Optimize both Occupations and Orbitals
// Start with DFT orbitals and Uniform Occupations
void optimize_full(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Full RDMFT Optimization (Occ + Orb) ---\n";
    
    int Na = 1; int Nb = 1;
    double target_Na = (double)Na; 
    double target_Nb = (double)Nb;

    // 1. Run DFT SCF to get initial orbitals
    std::cout << "\n>>> Running DFT SCF to get initial orbitals...\n";
    // Using LDA-like params
    int x_func_id = 1; // Slater
    int c_func_id = 7; // VWN
    auto orbits_dft = perform_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca = orbits_dft.first;
    arma::mat Cb = orbits_dft.second;

    int Na_orb = Ca.n_cols;
    int Nb_orb = Cb.n_cols;

    double power = 1.0; // HF-like
    auto func = std::make_shared<TestRDMFTFunctional>(basis, H0, Na_orb, power);
    Solver solver(func, S);

    // Combine C
    arma::mat C_tot(Ca.n_rows, Na_orb + Nb_orb);
    C_tot.cols(0, Na_orb - 1) = Ca;
    C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb;

    // Uniform occupations with noise to break symmetry
    arma::vec na(Na_orb); na.fill(target_Na / double(Na_orb));
    arma::vec nb(Nb_orb); nb.fill(target_Nb / double(Nb_orb));

    // Simple deterministic noise to break symmetry
    for(int i=0; i<Na_orb; ++i) {
        na(i) *= (1.0 + 0.1 * std::sin(i * 2.5));
        nb(i) *= (1.0 + 0.1 * std::cos(i * 3.5));
    }
    // Re-normalize roughly
    na = na * (target_Na / arma::sum(na));
    nb = nb * (target_Nb / arma::sum(nb));

    arma::vec n_tot(Na_orb + Nb_orb);
    n_tot.head(Na_orb) = na;
    n_tot.tail(Nb_orb) = nb;

    std::cout << "\nInitial Occupations (Perturbed Uniform): " << n_tot.head(5).t();

    solver.set_verbose(true);
    solver.set_optimize_occupations(true); 
    solver.set_optimize_orbitals(true);
    solver.set_max_outer_iter(100);
    solver.set_max_occ_iter(10);
    solver.set_max_orb_iter(50);
    solver.set_occ_tol(1e-7);
    solver.set_orb_tol(1e-8);
    
    // Orbital optimizer settings
    solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::CG);
    solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::MoreThuente);

    solver.solve(C_tot, n_tot, target_Na, target_Nb, Na_orb);

    std::cout << std::setprecision(12) << std::scientific;
    std::cout << "Final Energy: " << func->accumulated_energy << "\n";
    std::cout << std::defaultfloat << std::setprecision(6);
    
    std::cout << "Occupations Alpha (first 5): " << n_tot.head(5).t();
    std::cout << "Occupations Beta  (first 5): " << n_tot.subvec(Na_orb, Na_orb+4).t();
}

int main() {
    std::cout << "Testing RDMFT Solver: Full Optimization from DFT/Uniform\n";
    
    // Setup Basis (He atom)
    int Z = 2; 
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

    optimize_full(basis, S, H0);
    
    return 0;
}
