#include "rdmft_solver.h"
#include "general/scf_helpers.h"
#include "rdmft_gradients.h"
#include <iostream>
#include <algorithm> // For std::max, std::min
#include <cmath>     // For std::abs, std::isnan

namespace helfem {
namespace rdmft {

RDMFT_Solver::RDMFT_Solver(std::shared_ptr<EnergyFunctional<void>> functional, const arma::mat& S)
    : functional_(functional), S_(S) {
    
    // Compute S^{-1/2} (Sinvh)
    S_inv_sqrt_ = helfem::scf::form_Sinvh(S_, false);
    
    // Compute S^{1/2} = S * S^{-1/2}
    S_sqrt_ = S_ * S_inv_sqrt_;
}

arma::mat RDMFT_Solver::to_orthogonal_basis(const arma::mat& C) {
    return S_sqrt_.t() * C; // Ideally S_sqrt is symmetric, so S_sqrt * C
}

arma::mat RDMFT_Solver::from_orthogonal_basis(const arma::mat& X) {
    return S_inv_sqrt_ * X;
}

void RDMFT_Solver::solve(arma::mat& C, arma::vec& n, double target_N) {
    // Single channel backward compatibility wrapper
    solve(C, n, target_N, 0.0, -1);
}

void RDMFT_Solver::solve(arma::mat& C, arma::vec& n, double target_Na, double target_Nb, int n_alpha_orb) {
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

void RDMFT_Solver::optimize_occupations(arma::mat& C, arma::vec& n, double target_Na, double target_Nb, int n_alpha_orb, double& mu_a, double& mu_b, double rho) {
    // Projected Gradient Descent with simple line search
    
    // Initial energy
    arma::mat gC; arma::vec gn;
    double E = functional_->energy(C, n, gC, gn);
    
    for (int iter = 0; iter < max_occ_iter_; ++iter) {
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

void RDMFT_Solver::optimize_orbitals(arma::mat& C, arma::vec& n, int n_alpha_orb) {
    // Riemannian CG on Stiefel Manifold(s)
    
    // Switch to orthogonal basis
    arma::mat X = to_orthogonal_basis(C); 
    
    // Current Gradients
    arma::mat gC; arma::vec gn_eval;
    double E = functional_->energy(C, n, gC, gn_eval);
    
    // Gradient w.r.t X:  gX = S^{-1/2} gC
    arma::mat gX = S_inv_sqrt_ * gC; 
    
    // Riemannian gradient
    // If Two channels: We have two independent manifolds.
    // St(N_a, N_bf) x St(N_b, N_bf)
    // The metric is block diagonal.
    // grad_A = gX_A - X_A gX_A^T X_A
    // grad_B = gX_B - X_B gX_B^T X_B
    
    arma::mat grad(gX.n_rows, gX.n_cols);
    
    if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
         // Split
         arma::mat Xa = X.cols(0, n_alpha_orb - 1);
         arma::mat gXa = gX.cols(0, n_alpha_orb - 1);
         arma::mat grad_a = gXa - Xa * gXa.t() * Xa;
         
         arma::mat Xb = X.cols(n_alpha_orb, X.n_cols - 1);
         arma::mat gXb = gX.cols(n_alpha_orb, X.n_cols - 1);
         arma::mat grad_b = gXb - Xb * gXb.t() * Xb;
         
         grad.cols(0, n_alpha_orb - 1) = grad_a;
         grad.cols(n_alpha_orb, X.n_cols - 1) = grad_b;
    } else {
         grad = gX - X * gX.t() * X;
    }

    arma::mat dir = -grad;
    
    for (int iter = 0; iter < max_orb_iter_; ++iter) {
        
        double grad_norm = arma::norm(grad, "fro");
        if (grad_norm < orb_tol_) break;
        
        double step = 0.5; // Initial guess
        bool accepted = false;
        
        arma::mat X_new;
        arma::mat C_new;
        double E_new = E;
        
        // Backtracking line search
        int ls_iter = 0;
        while (ls_iter < 8) {
             // Retract
             arma::mat X_trial = X + step * dir;
             
             // Project back to Stiefel (QR)
             // Must be done per block if separate channels
             
             arma::mat Q, R;
             
             if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
                  // Block A
                  arma::mat Xta = X_trial.cols(0, n_alpha_orb - 1);
                  bool sta = arma::qr(Q, R, Xta);
                  if(!sta) X_new = X_trial; // fail
                  else {
                      // Fix signs A
                      arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                      for(arma::uword i=0; i<min_dim; ++i) if(R(i,i)<0) Q.col(i) *= -1.0;
                      X_new = X_trial; // initialize size
                      X_new.cols(0, n_alpha_orb - 1) = Q.cols(0, Xta.n_cols - 1);
                  }
                  
                  // Block B
                  arma::mat Xtb = X_trial.cols(n_alpha_orb, X.n_cols - 1);
                  bool stb = arma::qr(Q, R, Xtb);
                  if(!stb) { /* fail */ }
                  else {
                      // Fix signs B
                      arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                      for(arma::uword i=0; i<min_dim; ++i) if(R(i,i)<0) Q.col(i) *= -1.0;
                      X_new.cols(n_alpha_orb, X.n_cols - 1) = Q.cols(0, Xtb.n_cols - 1);
                  }
                  
             } else {
                  // Single block
                  bool status = arma::qr(Q, R, X_trial);
                  if (!status) { X_new = X; }
                  else {
                     arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                     for(arma::uword i=0; i<min_dim; ++i) {
                         if(R(i,i) < 0) Q.col(i) *= -1.0;
                     }
                     X_new = Q.cols(0, X.n_cols - 1);
                  }
             }
             
             C_new = from_orthogonal_basis(X_new);
             arma::mat gC_dummy; arma::vec gn_dummy;
             E_new = functional_->energy(C_new, n, gC_dummy, gn_dummy);
             
             if (E_new < E - 1e-4 * step * grad_norm*grad_norm) { 
                 accepted = true;
                 break;
             }
             step *= 0.5;
             ls_iter++;
        }
        
        if (!accepted) {
            if (verbose_) std::cout << "  [Orb] Line search failed or small step.\n";
            break; 
        }
        
        X = X_new;
        C = C_new;
        E = E_new;
        
        arma::mat gC_new; arma::vec gn_new;
        E = functional_->energy(C, n, gC_new, gn_new);
        arma::mat gX_new = S_inv_sqrt_ * gC_new;
        
        // Compute new gradient per block
        arma::mat grad_new(gX.n_rows, gX.n_cols);
        
        if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
             arma::mat Xa = X.cols(0, n_alpha_orb - 1);
             arma::mat gXa = gX_new.cols(0, n_alpha_orb - 1);
             arma::mat grad_a = gXa - Xa * gXa.t() * Xa;
             
             arma::mat Xb = X.cols(n_alpha_orb, X.n_cols - 1);
             arma::mat gXb = gX_new.cols(n_alpha_orb, X.n_cols - 1);
             arma::mat grad_b = gXb - Xb * gXb.t() * Xb;
             
             grad_new.cols(0, n_alpha_orb - 1) = grad_a;
             grad_new.cols(n_alpha_orb, X.n_cols - 1) = grad_b;
        } else {
             grad_new = gX_new - X * gX_new.t() * X;
        }
        
        // Steepest Descent reset for simplicity
        dir = -grad_new;
        grad = grad_new;
    }
    
    if (verbose_) std::cout << "  [Orb] Energy after optimize: " << E << "\n";
}

} // namespace rdmft
} // namespace helfem
