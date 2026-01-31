#include <algorithm>
#include "test_utils.h"

// Optimize Occupations Only (Neon)
void optimize_occ_only(helfem::atomic::basis::TwoDBasis& basis, const arma::mat& S, const arma::mat& H0) {
    std::cout << "\n--- Occupation Optimization Only (Neon) ---\n";

    // For Ne, alpha=5, beta=5
    int Na = 5; int Nb = 5;

    // Use DFT SCF to get reasonable initial orbitals
    int x_func_id = 1; int c_func_id = 7;
    auto orbits = perform_scf(basis, S, H0, Na, Nb, x_func_id, c_func_id);
    arma::mat Ca = orbits.first;
    arma::mat Cb = orbits.second;

    int Na_orb = Ca.n_cols;
    int Nb_orb = Cb.n_cols;

    // Use Goedecker-Umrigar (Gu) RDMFT functional (power=0.5)
    auto func = std::make_shared<TestRDMFTFunctional>(basis, H0, Na_orb, 0.5, XCFunctionalType::GoedeckerUmrigar);
    Solver solver(func, S);

    // Build concatenated C and occupations
    arma::mat C_tot(Ca.n_rows, Na_orb + Nb_orb);
    C_tot.cols(0, Na_orb - 1) = Ca;
    C_tot.cols(Na_orb, Na_orb + Nb_orb - 1) = Cb;

    // Initial occupations: uniform distribution across orbitals
    arma::vec na(Na_orb);
    arma::vec nb(Nb_orb);
    na.fill(double(Na) / double(Na_orb));
    nb.fill(double(Nb) / double(Nb_orb));

    arma::vec n_tot(Na_orb + Nb_orb);
    n_tot.head(Na_orb) = na;
    n_tot.tail(Nb_orb) = nb;

    std::cout << "Initial occupations (alpha head): " << n_tot.head(std::min(10, Na_orb)).t();

    solver.set_verbose(true);
    solver.set_max_outer_iter(150);
    solver.set_optimize_occupations(true); // OPTIMIZE OCCUPATIONS
    solver.set_optimize_orbitals(false);   // FIXED ORBITALS

    // Use default occupation optimizer (cosine param implementation)
    solver.solve(C_tot, n_tot, double(Na) + double(Nb));

    std::cout << "Occupation-Only Final Energy: " << func->accumulated_energy << "\n";
}

int main() {
    std::cout << "Testing RDMFT Solver: Occupation Optimization Only (Neon)\n";
    
    // Setup Basis
    int Z = 10; // Ne
    int primbas = 4; int Nnodes = 10; int Nelem = 6; double Rmax = 40.0;
    
    auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
    arma::vec bval = helfem::atomic::basis::form_grid((helfem::modelpotential::nuclear_model_t)0, 0.0, Nelem, Rmax, 
                                                      4, 2.0, Nelem, 4, 2.0, 
                                                      2, 0, 0, 0.0, false, 0.0);
    
    // Adding l=1 to support p-orbitals
    arma::ivec lvals(4); 
    arma::ivec mvals(4);
    
    int iang = 0;
    lvals(iang) = 0; mvals(iang) = 0; iang++;
    lvals(iang) = 1; mvals(iang) = 0; iang++;
    lvals(iang) = 1; mvals(iang) = 1; iang++;
    lvals(iang) = 1; mvals(iang) = -1; iang++;
    
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
