// test_occ_h2.cpp
// Occupation-only optimization for stretched H2 using HF orbitals

#include <iostream>
#include <armadillo>
#include <cmath>
#include "rdmft/solver.h"
#include "rdmft/energy.h"
#include "rdmft/gradients.h"
#include "general/scf_helpers.h"
#include "diatomic/basis.h"
#include "atomic/basis.h"
#include "test_utils.h"

using namespace helfem;
using namespace helfem::rdmft;

// Minimal test functional adapter for diatomic basis
struct TestRDMFTFunctionalDiatomic : public EnergyFunctional<void> {
    helfem::diatomic::basis::TwoDBasis& basis;
    arma::mat Hcore;
    double accumulated_energy;
    double power;
    int n_alpha_orb;
    XCFunctionalType xc_type;

    TestRDMFTFunctionalDiatomic(helfem::diatomic::basis::TwoDBasis& b, const arma::mat& H0, int na_orb, double p=1.0, XCFunctionalType type = XCFunctionalType::Power)
        : basis(b), Hcore(H0), accumulated_energy(0.0), power(p), n_alpha_orb(na_orb), xc_type(type) {}

    double energy(const arma::mat& C, const arma::vec& n) override {
        accumulated_energy = helfem::rdmft::compute_energy<helfem::diatomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, n_alpha_orb, xc_type);
        return accumulated_energy;
    }

    void orbital_gradient(const arma::mat& C, const arma::vec& n, arma::mat& gC) override {
        helfem::rdmft::compute_orbital_gradient<helfem::diatomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, gC, n_alpha_orb, xc_type);
    }

    void occupation_gradient(const arma::mat& C, const arma::vec& n, arma::vec& gn) override {
        helfem::rdmft::compute_occupation_gradient<helfem::diatomic::basis::TwoDBasis>(
            basis, Hcore, C, n, power, gn, n_alpha_orb, xc_type);
    }
};

int main() {
    std::cout << "Occupation-only optimization test: stretched H2+ (Müller p=0.5)\n";

    // Diatomic H2 parameters
    int Z1 = 1, Z2 = 1;
    double R = 6.0; // stretched bond (a0)
    double Rhalf = R / 2.0;

    int primbas = 4;
    int Nnodes = 6;
    int Nelem = 8;
    double Rmax = 30.0;

    auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
    arma::vec bval = helfem::atomic::basis::form_grid((helfem::modelpotential::nuclear_model_t)0, 0.0, Nelem, Rmax,
                                                      4, 2.0, Nelem, 4, 2.0, 1, 0, 0, Rhalf);

    arma::ivec lvals(1); lvals(0) = 0;
    arma::ivec mvals(1); mvals(0) = 0;

    helfem::diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, poly, 5*poly->get_nbf(), bval, lvals, mvals);
    basis.compute_tei(true);

    arma::mat S = basis.overlap();
    arma::mat T = basis.kinetic();
    arma::mat Vnuc = basis.nuclear();
    arma::mat H0 = T + Vnuc;

    // --- Diatomic HF setup: use total electron count ---
    int N_total = 1; // total electrons (H2+)
    int Na = (N_total + 1) / 2; // assign majority to alpha
    int Nb = N_total - Na;
    arma::mat Sinvh = basis.Sinvh(false, 0);

    arma::vec eps; arma::mat C;
    helfem::scf::eig_gsym(eps, C, H0, Sinvh);
    arma::mat Ca = C; arma::mat Cb = C;

    double E_old = 0.0;
    for (int iter=1; iter<=50; ++iter) {
        arma::mat Ca_occ;
        arma::mat Cb_occ;
        if (Na > 0) Ca_occ = Ca.cols(0, Na-1);
        else Ca_occ = arma::mat(Ca.n_rows, 0);
        if (Nb > 0) Cb_occ = Cb.cols(0, Nb-1);
        else Cb_occ = arma::mat(Cb.n_rows, 0);

        arma::mat Pa;
        if (Ca_occ.n_cols > 0) Pa = Ca_occ * Ca_occ.t();
        else Pa = arma::zeros(Ca.n_rows, Ca.n_rows);

        arma::mat Pb;
        if (Cb_occ.n_cols > 0) Pb = Cb_occ * Cb_occ.t();
        else Pb = arma::zeros(Cb.n_rows, Cb.n_rows);
        arma::mat Ptot = Pa + Pb;

        arma::mat J = basis.coulomb(Ptot);
        arma::mat Ka = basis.exchange(Pa);
        arma::mat Kb = basis.exchange(Pb);

        arma::mat Fa = H0 + J + Ka;
        arma::mat Fb = H0 + J + Kb;

        double E_core = arma::trace(Ptot * H0);
        double E_J = 0.5 * arma::trace(Ptot * J);
        double E_Exx = 0.5 * (arma::trace(Pa * Ka) + arma::trace(Pb * Kb));
        double E_tot = E_core + E_J + E_Exx;

        if (std::abs(E_tot - E_old) < 1e-8) break;
        E_old = E_tot;

        helfem::scf::eig_gsym(eps, Ca, Fa, Sinvh);
        helfem::scf::eig_gsym(eps, Cb, Fb, Sinvh);
    }

    // Assemble AO coefficient matrix (alpha then beta)
    int Norb = Ca.n_cols;
    arma::mat C_tot(Ca.n_rows, 2 * Norb);
    C_tot.cols(0, Norb-1) = Ca;
    C_tot.cols(Norb, 2*Norb-1) = Cb;

    // Initial uniform occupations
    arma::vec na0(Norb); na0.fill(double(Na) / double(Norb));
    arma::vec nb0(Norb); nb0.fill(double(Nb) / double(Norb));
    arma::vec n_tot(2*Norb);
    n_tot.head(Norb) = na0; n_tot.tail(Norb) = nb0;

    std::cout << "Initial occupations: " << n_tot.t();

    // Run occupation-only Müller (p=0.5)
    double power = 0.5;
    auto func = std::make_shared<TestRDMFTFunctionalDiatomic>(basis, H0, Norb, power);
    Solver solver(func, S);
    solver.set_verbose(true);
    solver.set_optimize_orbitals(false);
    solver.set_optimize_occupations(true);
    solver.set_max_occ_iter(300);
    solver.set_occ_tol(1e-8);

    solver.solve(C_tot, n_tot, double(Na+Nb));

    std::cout << "Final occupations: " << n_tot.t();

    // Report deviation from integer occupations (informational)
    arma::vec diff = arma::abs(n_tot - arma::round(n_tot));
    double maxdiff = arma::max(diff);
    std::cout << "Max deviation from integer occupations: " << maxdiff << std::endl;
    std::cout << "Finished H2 occupation-only test (inspect printed occupations above).\n";
    return 0;
}
