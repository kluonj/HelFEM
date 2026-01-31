#include "solver.h"
#include "general/scf_helpers.h"
#include "gradients.h"
#include <iostream>
#include <iomanip>
#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::abs, std::isnan

namespace helfem {
namespace rdmft {

Solver::Solver(std::shared_ptr<EnergyFunctional<void>> functional, const arma::mat& S)
    : functional_(functional), S_(S) {
    
    // Compute S^{-1/2} (Sinvh)
    S_inv_sqrt_ = helfem::scf::form_Sinvh(S_, false);
    
    // Compute S^{1/2} = S * S^{-1/2}
    S_sqrt_ = S_ * S_inv_sqrt_;
}

arma::mat Solver::to_orthogonal_basis(const arma::mat& C) {
    return S_sqrt_.t() * C; // Ideally S_sqrt is symmetric, so S_sqrt * C
}

arma::mat Solver::from_orthogonal_basis(const arma::mat& X) {
    return S_inv_sqrt_ * X;
}

void Solver::solve(arma::mat& C, arma::vec& n, double target_N) {
    if (verbose_) {
        std::cout << "Starting RDMFT Solver (Reimannian CG + Projected Gradient)\n";
        std::cout << "Target N=" << target_N << "\n";
    }

    double E_prev = 0.0;
    double mu = 0.0;
    double rho = 0.1; 

    for (int iter = 0; iter < max_outer_iter_; ++iter) {
        if (verbose_) std::cout << "Outer Iteration " << iter + 1 << "\n";
        
        // 1. Optimize Occupations
        if (do_optimize_occupations_) {
            optimize_occupations(C, n, target_N, mu, rho);
            if (verbose_) std::cout << "  [Occ] mu: " << mu << "\n";
        }
        
        // 2. Optimize Orbitals
        if (do_optimize_orbitals_) {
            optimize_orbitals(C, n);
        }

        // Check convergence 
        double E = functional_->energy(C, n);
        
        // Check nan
        if (std::isnan(E)) {
          if (verbose_) std::cout << "Energy is NaN! Stopping.\n";
          break;
        }

        double dE = std::abs(E - E_prev);
        if (verbose_) {
            std::cout << "  Total Energy: " << E << " deltaE: " << dE << "\n";
        }
        
        if (iter > 0 && dE <  orb_f_tol_ ) {
             if (verbose_) std::cout << "RDMFT converged.\n";
             break;
        }
        E_prev = E;
    }
}

void Solver::optimize_occupations(arma::mat& C, arma::vec& n, double target_N, double& mu, double rho) {
    // Projected Gradient Descent with simple line search
    
    // Initial energy
    double E = functional_->energy(C, n);
    arma::vec gn;
    functional_->occupation_gradient(C, n, gn);
    
    if (verbose_) {
        std::cout << "\n  --- Occupation Optimization ---\n";
        std::cout << "  Iter |      Energy      |   Grad Norm  |     mu     |   rho   | Occupations (first 8) ... \n";
        std::cout << "-------|------------------|--------------|------------|---------|---------------------------\n";
    }

    for (int iter = 1; iter < max_occ_iter_; ++iter) {
        
        if (verbose_) {
           std::cout << "  " << std::setw(4) << iter << " | " 
                     << std::scientific << std::setprecision(10) << std::setw(16) << E << " | " 
                     << std::scientific << std::setprecision(4) << std::setw(12) << arma::norm(gn, "inf") << " | "
                     << std::scientific << std::setprecision(4) << std::setw(10) << mu << " | "
                     << std::scientific << std::setprecision(4) << std::setw(7) << rho << " | ";
           
           // Print first few occ
           int n_print = std::min((int)n.n_elem, 8); 
           std::cout << std::fixed << std::setprecision(6);
           for(int k=0; k<n_print; ++k) std::cout << n(k) << " ";
           if(n.n_elem > (arma::uword)n_print) std::cout << "...";
           std::cout << "\n";
           std::cout.unsetf(std::ios::floatfield); // Reset formatting
        }

        double step = 1.0; 
        arma::vec n_new = n - step * gn;
        
        double current_la = 0.0;

        // Project
        n_new = project_capped_simplex(n_new, target_N, &current_la);
        
        // Backtracking
        double E_new = 0.0;
        int linesearch_steps = 0;
        bool step_accepted = false;
        int max_ls = 20;

          // Armijo-style backtracking with projection: require sufficient decrease
          const double c_armijo = 1e-4;
          while(linesearch_steps < max_ls) {
                 E_new = functional_->energy(C, n_new);
                 double slope = arma::dot(gn, n_new - n); // directional derivative after projection

                 // If slope >= 0 descent not guaranteed; reduce step and retry
                 if (slope < 0.0) {
                    if (E_new <= E + c_armijo * slope) {
                      step_accepted = true;
                      break; // Accept step
                    }
                 }

                 // Reduce step and recompute candidate
                 step *= 0.5;
                 n_new = n - step * gn;

                 // Project again inside LS
                 n_new = project_capped_simplex(n_new, target_N, &current_la);

                 linesearch_steps++;
          }
        
        if (!step_accepted) {
            if (verbose_) std::cout << "  [Occ] Line search failed to find lower energy. Stopping.\n";
            break;
        }
        
        double diff_n = arma::norm(n_new - n, "inf");
        n = n_new;
        E = E_new;
        
        // Update Lagrange multipliers
        mu = current_la / step;

        functional_->occupation_gradient(C, n, gn); 
        if (diff_n < occ_f_tol_) break;
    }
    
    if (verbose_) std::cout << "  [Occ] Energy after optimize: " << E << "\n";
}

void Solver::optimize_orbitals(arma::mat& C, arma::vec& n) {
    optimizer_.set_max_iter(max_orb_iter_);
    optimizer_.set_tol(orb_grad_tol_);
    optimizer_.set_verbose(verbose_);
    optimizer_.optimize(functional_, S_sqrt_, S_inv_sqrt_, C, n);
}

} // namespace rdmft
} // namespace helfem
