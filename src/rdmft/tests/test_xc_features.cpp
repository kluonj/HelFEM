#include "test_utils.h"
#include "general/diis.h"
#include <iostream>

int main() {
    std::cout << "Testing XC Functional Types..." << std::endl;

    // 1. Setup Basis (Lithium-like setup)
    int Z = 3;
    
    // Manual construction of TwoDBasis as per atomic/main.cpp or test_energy_gradient.cpp
    int primbas = 4;
    int Nnodes = 6;
    int Nelem = 10;
    // int lmax = 1, mmax = 1; // Support s and p orbitals
    double Rmax = 40.0;
    
    // We need to fetch the polynomial basis. 
    // Assuming helfem::polynomial_basis is available (included via TwoDBasis.h or similar)
    auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
    
    // Form grid
    arma::vec bval = helfem::atomic::basis::form_grid((helfem::modelpotential::nuclear_model_t)0, 0.0, Nelem, Rmax, 4, 2.0, Nelem, 4, 2.0, 2, 0, 0, 0.0, false, 0.0);
    
    // l-values and m-values. Support s and p orbitals.
    arma::ivec lvals(2); lvals(0)=0; lvals(1)=1;
    arma::ivec mvals(2); mvals(0)=0; mvals(1)=0;

    helfem::atomic::basis::TwoDBasis basis(Z, (helfem::modelpotential::nuclear_model_t)0, 0.0, poly, false, 5*poly->get_nbf(), bval, poly->get_nprim()-1, lvals, mvals, 0, 0, 0.0);
    
    // 2. Setup Hcore
    basis.compute_tei(true); // Initialize TEI machinery
    
    arma::mat S = basis.overlap();
    arma::mat T = basis.kinetic();
    arma::mat V = basis.nuclear();
    arma::mat Hcore = T + V;
    
    // 3. Perform HF SCF for Li+ (2 electrons) to get converged orbitals
    std::cout << "Running SCF for Li+ to get converged orbitals..." << std::endl;
    
    // Initial guess from core Hamiltonian
    arma::mat Sinvh = helfem::scf::form_Sinvh(S, /*chol=*/false);
    arma::vec eps; arma::mat evec;
    arma::mat Horth = arma::trans(Sinvh) * Hcore * Sinvh;
    arma::eig_sym(eps, evec, Horth);
    evec = Sinvh * evec;
    
    int nela = 1;
    int nelb = 1;
    
    arma::mat Ca = evec; 
    arma::mat Caocc = Ca.cols(0, nela-1);
    arma::mat Cbocc = Caocc; // Restricted
    
    uDIIS diis(S, Sinvh, false, true, 1e-2, 1e-3, true, true, 5);
    double Etot = 0.0, Eold = 0.0, diiserr=0.0;
    int maxit = 50;
    double convthr = 1e-9;
    
    for(int it=1; it<=maxit; ++it) {
        arma::mat Pa = helfem::scf::form_density(Caocc, nela);
        arma::mat Pb = helfem::scf::form_density(Cbocc, nelb);
        arma::mat P = Pa + Pb; // Total density matrix

        arma::mat J = basis.coulomb(P);
        arma::mat Ka = basis.exchange(Pa);
        arma::mat Kb = basis.exchange(Pb);

        double Ekin = arma::trace(P * T);
        double Epot = arma::trace(P * V);
        double Ecoul = 0.5 * arma::trace(P * J);
        double Exx = 0.5 * arma::trace(Pa * Ka) + 0.5 * arma::trace(Pb * Kb);
        Etot = Ekin + Epot + Ecoul + Exx;

        arma::mat Fa = Hcore + J + Ka;
        arma::mat Fb = Hcore + J + Kb;

        diis.update(Fa, Fb, Pa, Pb, Etot, diiserr);
        diis.solve_F(Fa, Fb);

        arma::vec Ea_eval; arma::mat Ca_new;
        helfem::scf::eig_gsym(Ea_eval, Ca_new, Fa, Sinvh);
        Ca = Ca_new; 
        Caocc = Ca.cols(0, nela-1);
        Cbocc = Caocc;

        double dE = std::abs(Etot - Eold);
        if(dE < convthr) {
            std::cout << "SCF Converged at it=" << it << " E=" << Etot << std::endl;
            break;
        }
        Eold = Etot;
    }

    // 4. Setup converged C and n
    int Norb = basis.Nbf();
    arma::vec n(2*Norb, arma::fill::zeros);
    // Closed shell Li+ : lowest orbital has occupation 1.0 per spin
    n(0) = 1.0; 
    n(Norb) = 1.0;
    
    arma::mat C = Ca; 
    
    // 5. Test Muller (Power=0.5)
    // Note: TestRDMFTFunctional constructor: (basis, H0, n_alpha, power, type)
    TestRDMFTFunctional func_muller(basis, Hcore, 0, 0.5, XCFunctionalType::Muller);
    arma::mat gC; arma::vec gn;
    double E_muller = func_muller.energy(C, n, gC, gn);
    std::cout << "Muller Energy: " << E_muller << std::endl;
    
    // 6. Test HF (Power=1.0)
    TestRDMFTFunctional func_hf(basis, Hcore, 0, 1.0, XCFunctionalType::HartreeFock);
    double E_hf = func_hf.energy(C, n, gC, gn);
    std::cout << "HF Energy: " << E_hf << std::endl;
    
    if (std::abs(E_muller - E_hf) < 1e-6) {
        std::cout << "Note: Muller and HF energies are identical. This might happen for single orbital systems if functional form reduces to same." << std::endl;
    }
    
    // 7. Test GU (Muller + Correction)
    TestRDMFTFunctional func_gu(basis, Hcore, 0, 0.5, XCFunctionalType::GoedeckerUmrigar);
    double E_gu = func_gu.energy(C, n, gC, gn);
    std::cout << "GU Energy: " << E_gu << std::endl;
    
    // 8. Test Fractional Occupations (Verify Sensitivity)
    std::cout << "Testing Fractional Occupations..." << std::endl;
    arma::vec n_frac = n;
    // Move some population to excited state
    if (Norb > 1) {
        n_frac(0) = 0.9; n_frac(1) = 0.1; 
        n_frac(Norb) = 0.9; n_frac(Norb+1) = 0.1;
        
        double E_mul_frac = func_muller.energy(C, n_frac, gC, gn);
        double E_hf_frac = func_hf.energy(C, n_frac, gC, gn);
        
        std::cout << "Fractional Muller: " << E_mul_frac << std::endl;
        std::cout << "Fractional HF: " << E_hf_frac << std::endl;
        
        if (std::abs(E_mul_frac - E_hf_frac) > 1e-6) {
             std::cout << "SUCCESS: Functionals differ for fractional occupations." << std::endl;
        } else {
             std::cout << "FAILURE: Functionals should differ for fractional occupations!" << std::endl;
             return 1;
        }
    } else {
        std::cout << "Skipping fractional test (Norb < 2)." << std::endl;
    }

    std::cout << "XC Features Test Passed!" << std::endl;
    
    return 0;
}
