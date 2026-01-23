#include <algorithm>
#include "test_utils.h"

// Optimize Orbitals Only (Starting from H0 diagonalization)
void optimize_orb_only_h0(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Orbital Optimization Only (H0 Guess, Fixed Occupations) ---\n";

    // For He, alpha=1, beta=1
    int Na = 1; int Nb = 1;

    // Diagonalize H0 to get guess orbitals
    // H C = S C E => H0 C = S C E
    // Use Sinvh = S^{-1/2}
    std::cout << "Diagonalizing H0 for initial guess orbitals...\n";
    arma::mat Sinvh = helfem::scf::form_Sinvh(S, false);
    arma::vec eps; 
    arma::mat C_guess;
    
    // Hguess is just H0 for bare Coulomb
    helfem::scf::eig_gsym(eps, C_guess, H0, Sinvh);
    
    int Na_orb = C_guess.n_cols;
    int Nb_orb = Na_orb;

    // Use C_guess for both
    arma::mat Ca = C_guess;
    arma::mat Cb = C_guess; // Initial guess same for alpha/beta

    auto func = std::make_shared<TestRDMFTFunctional>(basis, H0, Na_orb);
    Solver solver(func, S);

    // Build concatenated C and occupations
    arma::mat C_tot(Ca.n_rows, Na_orb + Nb_orb);
    C_tot.cols(0, Na_orb - 1) = Ca;
    C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb;

    // Fixed occupations: put all electrons into lowest orbitals (integer occupations)
    arma::vec na(Na_orb, arma::fill::zeros);
    arma::vec nb(Nb_orb, arma::fill::zeros);
    
    // Explicitly set integer occupations for He
    na(0) = 1.0; 
    nb(0) = 1.0;

    arma::vec n_tot(Na_orb + Nb_orb);
    n_tot.head(Na_orb) = na;
    n_tot.tail(Nb_orb) = nb;

    std::cout << "Initial occupations (alpha head): " << n_tot.head(std::min(5, Na_orb)).t();

    solver.set_verbose(true);
    solver.set_max_outer_iter(200);

    solver.set_optimize_occupations(false); // FIXED OCCUPATIONS
    solver.set_optimize_orbitals(true);    // OPTIMIZE ORBITALS
    solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::CG);
    solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::MoreThuente);
    solver.set_orbital_lbfgs_history(8);
    solver.set_orbital_preconditioner(helfem::rdmft::OrbitalOptimizer::Preconditioner::None);

    // Use dual-channel solve entrypoint
    solver.solve(C_tot, n_tot, double(Na), double(Nb), Na_orb);

    std::cout << "Orbital-Only Final Energy: " << func->accumulated_energy << "\n";
}

int main() {
    std::cout << "Testing RDMFT Solver: Orbital Optimization Only (H0 Guess)\n";
    
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

    optimize_orb_only_h0(basis, S, H0);
    
    return 0;
}
