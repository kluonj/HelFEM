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
    // Single channel backward compatibility wrapper
    solve(C, n, target_N, 0.0, -1);
}

void Solver::solve(arma::mat& C, arma::vec& n, double target_Na, double target_Nb, int n_alpha_orb) {
    if (verbose_) {
        std::cout << "Starting RDMFT Solver (Reimannian CG + Projected Gradient)\n";
        if(n_alpha_orb > 0) std::cout << "Dual Channel Mode: Na=" << target_Na << ", Nb=" << target_Nb << "\n";
        else std::cout << "Single Channel Mode: N=" << target_Na << "\n";
    }

    double E_prev = 0.0;
    double mu_a = 0.0;
    double mu_b = 0.0;
    double rho = 0.1; 

    for (int iter = 0; iter < max_outer_iter_; ++iter) {
        if (verbose_) std::cout << "Outer Iteration " << iter + 1 << "\n";
        
        // 1. Optimize Occupations
        if (do_optimize_occupations_) {
            optimize_occupations(C, n, target_Na, target_Nb, n_alpha_orb, mu_a, mu_b, rho);
            if (verbose_) {
              if (n_alpha_orb > 0) std::cout << "  [Occ] mu_a: " << mu_a << " mu_b: " << mu_b << "\n";
              else std::cout << "  [Occ] mu: " << mu_a << "\n"; // In single channel, mu_a holds the value
            }
        }
        
        // 2. Optimize Orbitals
        if (do_optimize_orbitals_) {
            optimize_orbitals(C, n, n_alpha_orb);
        }

        // Check convergence 
        arma::mat gC; arma::vec gn;
        double E = functional_->energy(C, n, gC, gn);
        
        // Check nan
        if (std::isnan(E)) {
          if (verbose_) std::cout << "Energy is NaN! Stopping.\n";
          break;
        }

        double dE = std::abs(E - E_prev);
        if (verbose_) {
            std::cout << "  Total Energy: " << E << " deltaE: " << dE << "\n";
        }
        
        if (iter > 0 && dE < std::max(occ_tol_, orb_tol_) && dE < 1e-7) {
             if (verbose_) std::cout << "RDMFT converged.\n";
             break;
        }
        E_prev = E;
    }
}

void Solver::optimize_occupations(arma::mat& C, arma::vec& n, double target_Na, double target_Nb, int n_alpha_orb, double& mu_a, double& mu_b, double rho) {
    // Projected Gradient Descent with simple line search
    
    // Initial energy
    arma::mat gC; arma::vec gn;
    double E = functional_->energy(C, n, gC, gn);
    
    if (verbose_) {
        std::cout << "\n  --- Occupation Optimization ---\n";
        std::cout << "  Iter |      Energy      |   Grad Norm  |     mu     |   rho   | Occupations (first 8) ... \n";
        std::cout << "-------|------------------|--------------|------------|---------|---------------------------\n";
    }

    for (int iter = 1; iter < max_occ_iter_; ++iter) {
        
        if (verbose_) {
           std::cout << "  " << std::setw(4) << iter << " | " 
                     << std::scientific << std::setprecision(8) << std::setw(16) << E << " | " 
                     << std::scientific << std::setprecision(4) << std::setw(12) << arma::norm(gn, "inf") << " | "
                     << std::scientific << std::setprecision(4) << std::setw(10) << mu_a << " | "
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
        double current_lb = 0.0;

        // Project
        if (n_alpha_orb > 0 && n.n_elem > (arma::uword)n_alpha_orb) {
            // Split projection
            // Na part
            arma::vec na = n_new.head(n_alpha_orb);
            na = project_capped_simplex(na, target_Na, &current_la);
            
            // Nb part
            arma::vec nb = n_new.tail(n_new.n_elem - n_alpha_orb);
            nb = project_capped_simplex(nb, target_Nb, &current_lb);
                
            // Reassemble
            n_new.head(n_alpha_orb) = na;
            n_new.tail(n_new.n_elem - n_alpha_orb) = nb;
        } else {
            // Single channel
            n_new = project_capped_simplex(n_new, target_Na, &current_la);
            current_lb = current_la;
        }
        
        // Backtracking
        double E_new = 0.0;
        int linesearch_steps = 0;
        while(linesearch_steps < 5) {
             arma::mat gC_dummy; arma::vec gn_dummy;
             E_new = functional_->energy(C, n_new, gC_dummy, gn_dummy);
             if (E_new < E + 1e-10) break; // Accept step
             
             step *= 0.5;
             n_new = n - step * gn;
             
             // Project again inside LS
             if (n_alpha_orb > 0 && n.n_elem > (arma::uword)n_alpha_orb) {
                arma::vec na = n_new.head(n_alpha_orb);
                na = project_capped_simplex(na, target_Na, &current_la);
                arma::vec nb = n_new.tail(n_new.n_elem - n_alpha_orb);
                nb = project_capped_simplex(nb, target_Nb, &current_lb);
                n_new.head(n_alpha_orb) = na;
                n_new.tail(n_new.n_elem - n_alpha_orb) = nb;
             } else {
                n_new = project_capped_simplex(n_new, target_Na, &current_la);
                current_lb = current_la;
             }
             
             linesearch_steps++;
        }
        
        double diff_n = arma::norm(n_new - n, "inf");
        n = n_new;
        E = E_new;
        
        // Update Lagrange multipliers
        mu_a = current_la / step;
        mu_b = current_lb / step;

        functional_->energy(C, n, gC, gn); 
        if (diff_n < occ_tol_) break;
    }
    
    if (verbose_) std::cout << "  [Occ] Energy after optimize: " << E << "\n";
}

void Solver::optimize_orbitals(arma::mat& C, arma::vec& n, int n_alpha_orb) {
    // Riemannian CG on Stiefel Manifold(s)
    
    // Switch to orthogonal basis
    arma::mat X = to_orthogonal_basis(C); 
    
    // Current Gradients
    arma::mat gC; arma::vec gn_eval;
    double E = functional_->energy(C, n, gC, gn_eval);
    
    // Gradient w.r.t X:  gX = S^{-1/2} gC
    arma::mat gX_Euc = S_inv_sqrt_ * gC; 
    
    // Helper to calc Riemannian grad from gX_Euc and X
    auto calc_riem_grad = [&](const arma::mat& X_in, const arma::mat& gX_in) {
         arma::mat G(X_in.n_rows, X_in.n_cols);
         if (n_alpha_orb > 0 && X_in.n_cols > (arma::uword)n_alpha_orb) {
             arma::mat Xa = X_in.cols(0, n_alpha_orb - 1);
             arma::mat gXa = gX_in.cols(0, n_alpha_orb - 1);
             arma::mat ga = gXa - Xa * gXa.t() * Xa;
             G.cols(0, n_alpha_orb - 1) = ga;
             
             arma::mat Xb = X_in.cols(n_alpha_orb, X_in.n_cols - 1);
             arma::mat gXb = gX_in.cols(n_alpha_orb, X_in.n_cols - 1);
             arma::mat gb = gXb - Xb * gXb.t() * Xb;
             G.cols(n_alpha_orb, X_in.n_cols - 1) = gb;
         } else {
             G = gX_in - X_in * gX_in.t() * X_in;
         }
         return G;
    };
    
    arma::mat grad = calc_riem_grad(X, gX_Euc);
    arma::mat dir = -grad;
    
    if (verbose_) {
        std::cout << "\n  --- Orbital Optimization ---\n";
        std::cout << "  Iter |      Energy      | Riem Grad Norm |  Step Size \n";
        std::cout << "-------|------------------|----------------|------------\n";
    }

    for (int iter = 1; iter < max_orb_iter_; ++iter) {
        
        double grad_norm = arma::norm(grad, "fro");
        if (grad_norm < orb_tol_) break;

        if (verbose_) {
             std::cout << "  " << std::setw(4) << iter << " | "
                       << std::scientific << std::setprecision(8) << std::setw(16) << E << " | "
                       << std::scientific << std::setprecision(4) << std::setw(14) << grad_norm << " | " << std::flush;
        }
        
        arma::mat C_new, X_new;
        double dphi_0 = arma::dot(grad, dir);
        if (dphi_0 >= 0) {
             dir = -grad;
             dphi_0 = -arma::dot(grad, grad);
        }

        double step = perform_linesearch(C, n, X, dir, E, dphi_0, n_alpha_orb, C_new, X_new);
        
        if (verbose_) {
             std::cout << std::scientific << std::setprecision(4) << std::setw(10) << step << "\n";
        }

        if (step == 0.0) {
            if (verbose_) std::cout << "  [Orb] Line search failed or small step.\n";
            // Reduce trust region or restart? For now break.
            break; 
        }
        
        // Compute new gradient
        arma::mat gC_new; arma::vec gn_new;
        E = functional_->energy(C_new, n, gC_new, gn_new);
        arma::mat gX_new_Euc = S_inv_sqrt_ * gC_new;
        arma::mat grad_new = calc_riem_grad(X_new, gX_new_Euc);
        
        // Polak-Ribiere CG
        // Transport grad to new point
        arma::mat Xt_g = X_new.t() * grad;
        arma::mat sym_Xt_g = 0.5 * (Xt_g + Xt_g.t());
        arma::mat grad_trans = grad - X_new * sym_Xt_g;
        
        double num = arma::dot(grad_new, grad_new - grad_trans);
        double den = arma::dot(grad, grad);
        double beta = std::max(0.0, num / den);
        
        // Transport dir to new point
        arma::mat Xt_d = X_new.t() * dir;
        arma::mat sym_Xt_d = 0.5 * (Xt_d + Xt_d.t());
        arma::mat dir_trans = dir - X_new * sym_Xt_d;
        
        arma::mat dir_new = -grad_new + beta * dir_trans;
        
        // Restart if not descent
        if (arma::dot(grad_new, dir_new) >= 0) {
            dir_new = -grad_new;
        }
        
        X = X_new;
        C = C_new;
        grad = grad_new;
        dir = dir_new;
    }
    
    if (verbose_) std::cout << "  [Orb] Energy after optimize: " << E << "\n";
}

double Solver::perform_linesearch(const arma::mat& C, const arma::vec& n, const arma::mat& X, const arma::mat& dir, double E_initial, double dphi_0, int n_alpha_orb, arma::mat& C_new, arma::mat& X_new) {
    // Backtracking Armijo Line Search
    // Robust for Steepest Descent
    
    double alpha = 1.0;
    double rho = 0.5;
    double c1 = 1e-4;
    int max_ls = 20;

    for(int i=0; i<max_ls; ++i) {
         // Retract X -> X_new
         arma::mat X_trial = X + alpha * dir;
         arma::mat X_out;

         // QR Retraction
         if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
              // Block A
              arma::mat Xta = X_trial.cols(0, n_alpha_orb - 1);
              arma::mat Q, R;
              if(!arma::qr(Q, R, Xta)) { X_out = X_trial; }
              else {
                  arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                  for(arma::uword k=0; k<min_dim; ++k) if(R(k,k)<0) Q.col(k) *= -1.0;
                  X_out = X_trial; // Init size
                  X_out.cols(0, n_alpha_orb - 1) = Q.cols(0, Xta.n_cols - 1);
              }
              // Block B
              arma::mat Xtb = X_trial.cols(n_alpha_orb, X.n_cols - 1);
              if(arma::qr(Q, R, Xtb)) {
                  arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                  for(arma::uword k=0; k<min_dim; ++k) if(R(k,k)<0) Q.col(k) *= -1.0;
                  X_out.cols(n_alpha_orb, X.n_cols - 1) = Q.cols(0, Xtb.n_cols - 1);
              }
         } else {
              arma::mat Q, R;
              if (!arma::qr(Q, R, X_trial)) { X_out = X_trial; }
              else {
                 arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                 for(arma::uword k=0; k<min_dim; ++k) if(R(k,k) < 0) Q.col(k) *= -1.0;
                 X_out = Q.cols(0, X.n_cols - 1);
              }
         }
         
         arma::mat C_trial = from_orthogonal_basis(X_out);
         arma::mat gC_local; arma::vec gn_local;
         double E_new = functional_->energy(C_trial, n, gC_local, gn_local);
         
         // Armijo Condition
         if (E_new <= E_initial + c1 * alpha * dphi_0) {
             X_new = X_out;
             C_new = C_trial;
             return alpha;
         }
         
         alpha *= rho;
    }
    
    return 0.0;
}

} // namespace rdmft
} // namespace helfem
