#include <algorithm>
#include "test_utils.h"

// Wrapper to handle basis normalization
class NormalizedRDMFTFunctional : public helfem::rdmft::EnergyFunctional<void> {
public:
    std::shared_ptr<helfem::rdmft::EnergyFunctional<void>> inner_func;
    arma::vec d; // Normalization factors

    NormalizedRDMFTFunctional(std::shared_ptr<helfem::rdmft::EnergyFunctional<void>> func, const arma::vec& scale_factors)
        : inner_func(func), d(scale_factors) {
    }

    double energy(const arma::mat& C_norm, const arma::vec& n, arma::mat& gC_norm, arma::vec& gn) override {
        // C_unnorm = D * C_norm
        arma::mat C_unnorm = C_norm;
        C_unnorm.each_col() %= d;
        
        arma::mat gC_unnorm;
        double E = inner_func->energy(C_unnorm, n, gC_unnorm, gn);
        
        // Gradient transform: g_norm = D * g_unnorm
        gC_norm = gC_unnorm;
        gC_norm.each_col() %= d;
        
        return E;
    }
};

// Optimize Orbitals Only (Fixed Occupations from Integer Guess)
void optimize_orb_only(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Orbital Optimization Only (Fixed Occupations) ---\n";

    // For He, alpha=1, beta=1
    int Na = 1; int Nb = 1;

    // Use DFT SCF to get reasonable initial orbitals (Unnormalized basis)
    int x_func_id = 1; int c_func_id = 7;
    auto orbits = perform_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca = orbits.first;
    arma::mat Cb = orbits.second;

    int Na_orb = Ca.n_cols;
    int Nb_orb = Cb.n_cols;

    // --- Normalization Logic ---
    arma::vec d = 1.0 / arma::sqrt(arma::diagvec(S));
    arma::mat S_norm = S;
    S_norm.each_col() %= d; // S * D
    S_norm.each_row() %= d.t(); // D * S * D
    std::cout << "Normalized S diagonal range: " << S_norm.diag().min() << " - " << S_norm.diag().max() << "\n";

    // Create inner functional (works with unnormalized quantities)
    auto inner_func = std::make_shared<TestRDMFTFunctional>(basis, H0, Na_orb);
    auto func = std::make_shared<NormalizedRDMFTFunctional>(inner_func, d);
    
    // Solver uses normalized S
    Solver solver(func, S_norm);

    // Build concatenated C and occupations
    arma::mat C_tot(Ca.n_rows, Na_orb + Nb_orb);
    C_tot.cols(0, Na_orb - 1) = Ca;
    C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb;

    // Normalize Initial Guess C: C_norm = D^-1 * C_unnorm
    // Since C_unnorm = D * C_norm  =>  C_norm = D^-1 * C_unnorm
    C_tot.each_col() /= d;

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
    solver.set_max_outer_iter(100);
    solver.set_optimize_occupations(false); // FIXED OCCUPATIONS
    solver.set_optimize_orbitals(true);    // OPTIMIZE ORBITALS
    solver.set_max_orb_iter(100);
    // solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::LBFGS);
    solver.set_orbital_optimizer(helfem::rdmft::OrbitalOptimizer::Method::LBFGS);
    solver.set_orbital_linesearch(helfem::rdmft::OrbitalOptimizer::LineSearch::StrongWolfe);
    // solver.set_orbital_lbfgs_history(8);
    solver.set_orbital_preconditioner(helfem::rdmft::OrbitalOptimizer::Preconditioner::None);

    // Use dual-channel solve entrypoint
    solver.solve(C_tot, n_tot, double(Na), double(Nb), Na_orb);

    std::cout << "Orbital-Only Final Energy: " << inner_func->accumulated_energy << "\n";
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
