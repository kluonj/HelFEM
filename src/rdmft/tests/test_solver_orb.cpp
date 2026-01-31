#include <algorithm>
#include "test_utils.h"

// Optimize Orbitals Only (Fixed Occupations from Integer Guess)
void optimize_orb_only(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Orbital Optimization Only (Fixed Occupations) ---\n";

    // For He, alpha=1, beta=1
    int Na = 1; int Nb = 1;

    // Use DFT SCF to get reasonable initial orbitals
    int x_func_id = 1; int c_func_id = 7;
    auto orbits = perform_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca = orbits.first;
    arma::mat Cb = orbits.second;

    int Na_orb = Ca.n_cols;
    int Nb_orb = Cb.n_cols;

    // Use Restricted Spatial Mode: Pass n_alpha_orb=0 (so Functional detects 2*Norb occupations)
    auto func = std::make_shared<TestRDMFTFunctional>(basis, H0, 0);
    Solver solver(func, S);

    // Use only spatial orbitals (Restricted)
    arma::mat C_tot = Ca; 
    
    // Perturb C slightly? Not needed if starting from SCF.
    // C_tot = Ca + 0.01 * arma::randu(size(Ca)); -> Might break orthonormality. 
    // Solver will not re-orthogonalize automatically? 
    // Actually Solver assumes input C is "guess". 
    // But optimizer assumes "orthogonal basis" X = S^1/2 C.
    // The optimizer computes X from C. If we perturb C, X will be perturbed.
    // But optimizer projects grad to tangent. It doesn't project X to manifold until line search steps?
    // Wait, evaluate_point uses retract().
    // The initial point X_base comes from C.
    // If we want to guarantee X is on manifold, we should orthogonalize C first.
    // C = to_orthogonal(C)...
    // But Ca from SCF is already orthogonal.

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
    solver.set_max_outer_iter(50);
    solver.set_max_orb_iter(50);
    solver.set_optimize_occupations(false); // FIXED OCCUPATIONS
    solver.set_optimize_orbitals(true);    // OPTIMIZE ORBITALS
    // solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::LBFGS);
    solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::CG);
    solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::MoreThuente);
    // solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::Armijo);
    // solver.set_orbital_lbfgs_history(8);
    solver.set_orbital_preconditioner(helfem::rdmft::OrbitalOptimizer::Preconditioner::None);

    // Use dual-channel solve entrypoint
    solver.solve(C_tot, n_tot, double(Na) + double(Nb));

    std::cout << "Orbital-Only Final Energy: " << func->accumulated_energy << "\n";
}

int main() {
    std::cout << "Testing RDMFT Solver: Orbital Optimization Only\n";
    
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

    optimize_orb_only(basis, S, H0);
    
    return 0;
}
