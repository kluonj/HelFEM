#include "rdmft_solver.h"
#include "general/scf_helpers.h"
#include "rdmft_gradients.h"
#include <iostream>
#include <iomanip>
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
    
    if (verbose_) {
        std::cout << "\n  --- Occupation Optimization ---\n";
        std::cout << "  Iter |      Energy      |   Grad Norm  | Occupations (first 8) ... \n";
        std::cout << "-------|------------------|--------------|---------------------------\n";
    }

    for (int iter = 0; iter < max_occ_iter_; ++iter) {
        
        if (verbose_) {
           std::cout << "  " << std::setw(4) << iter << " | " 
                     << std::scientific << std::setprecision(8) << std::setw(16) << E << " | " 
                     << std::scientific << std::setprecision(4) << std::setw(12) << arma::norm(gn, "inf") << " | ";
           
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
        
        arma::mat C_new, X_new;
        // For steepest descent dir = -grad, dphi_0 = - ||grad||^2. 
        // If CG, dphi_0 = dot(grad, dir).
        double dphi_0 = arma::dot(grad, dir);

        double step = perform_linesearch(C, n, X, dir, E, dphi_0, n_alpha_orb, C_new, X_new);
        
        if (step == 0.0) {
            if (verbose_) std::cout << "  [Orb] Line search failed or small step.\n";
            break; 
        }
        
        X = X_new;
        C = C_new;
        
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

double RDMFT_Solver::perform_linesearch(const arma::mat& C, const arma::vec& n, const arma::mat& X, const arma::mat& dir, double E_initial, double dphi_0, int n_alpha_orb, arma::mat& C_new, arma::mat& X_new) {
    // Strong Wolfe Line Search with Zoom (Nocedal & Wright, Algorithm 3.5 & 3.6)
    
    // Config
    double c1 = 1e-4;
    double c2 = 0.9;
    double alpha_max = 2.0;
    double alpha_curr = 1.0; 
    double alpha_prev = 0.0;
    
    double phi_0 = E_initial;
    double phi_prev = phi_0;
    double dphi_prev = dphi_0;        
    // dphi_0 passed in

    // Helper: Evaluation Function
    // Computes phi(alpha) and dphi(alpha)
    // Also outputs the resulting C_new and X_new for success case
    auto eval_step = [&](double alpha, double& phi, double& dphi, arma::mat& X_out, arma::mat& C_out) {
         // Retract X -> X_new
         arma::mat X_trial = X + alpha * dir;
         
         // Project back to Stiefel (QR)
         arma::mat Q, R;
         if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
              // Block A
              arma::mat Xta = X_trial.cols(0, n_alpha_orb - 1);
              if(!arma::qr(Q, R, Xta)) { X_out = X_trial; }
              else {
                  arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                  for(arma::uword i=0; i<min_dim; ++i) if(R(i,i)<0) Q.col(i) *= -1.0;
                  X_out = X_trial; 
                  X_out.cols(0, n_alpha_orb - 1) = Q.cols(0, Xta.n_cols - 1);
              }
              // Block B
              arma::mat Xtb = X_trial.cols(n_alpha_orb, X.n_cols - 1);
              if(!arma::qr(Q, R, Xtb)) { /* fail */ }
              else {
                  arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                  for(arma::uword i=0; i<min_dim; ++i) if(R(i,i)<0) Q.col(i) *= -1.0;
                  X_out.cols(n_alpha_orb, X.n_cols - 1) = Q.cols(0, Xtb.n_cols - 1);
              }
         } else {
              // Single block
              if (!arma::qr(Q, R, X_trial)) { X_out = X; }
              else {
                 arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                 for(arma::uword i=0; i<min_dim; ++i) if(R(i,i) < 0) Q.col(i) *= -1.0;
                 X_out = Q.cols(0, X.n_cols - 1);
              }
         }
         
         C_out = from_orthogonal_basis(X_out);
         arma::mat gC_local; arma::vec gn_local;
         phi = functional_->energy(C_out, n, gC_local, gn_local);
         
         // Riemannian Grad at X_out
         arma::mat gX = S_inv_sqrt_ * gC_local;
         arma::mat grad_new;
         
         if (n_alpha_orb > 0 && X.n_cols > (arma::uword)n_alpha_orb) {
             arma::mat Xa = X_out.cols(0, n_alpha_orb - 1);
             arma::mat gXa = gX.cols(0, n_alpha_orb - 1);
             arma::mat grad_a = gXa - Xa * gXa.t() * Xa;
             
             arma::mat Xb = X_out.cols(n_alpha_orb, X.n_cols - 1);
             arma::mat gXb = gX.cols(n_alpha_orb, X.n_cols - 1);
             arma::mat grad_b = gXb - Xb * gXb.t() * Xb;
             
             grad_new = gX; // Init size
             grad_new.cols(0, n_alpha_orb - 1) = grad_a;
             grad_new.cols(n_alpha_orb, X.n_cols - 1) = grad_b;
         } else {
             grad_new = gX - X_out * gX.t() * X_out;
         }

         // Vector Transport of dir: T(eta) = eta - X_new * sym(X_new^T * eta)
         // Assuming Euclidean transport approx for step is roughly okay, 
         // but proper projection transport is better.
         arma::mat Xt_dir = X_out.t() * dir;
         arma::mat sym_Xt_dir = 0.5 * (Xt_dir + Xt_dir.t());
         arma::mat transported_dir = dir - X_out * sym_Xt_dir;

         dphi = arma::dot(grad_new, transported_dir);
    };

    // Zoom Function (Algorithm 3.6)
    auto zoom = [&](double alo, double ahi, double phi_lo, double phi_hi, double dphi_lo) -> double {
        int max_zoom = 10;
        for(int j=0; j<max_zoom; ++j) {
            // Interpolation (Cubic/Quadratic) - simplified to bisection/quadratic
            // Using simple bisection for safety for now, or quadratic
            double aj = 0.5 * (alo + ahi); 

            arma::mat Xj, Cj;
            double phij, dphij;
            eval_step(aj, phij, dphij, Xj, Cj);
            
            // Check Armijo
            if (phij > phi_0 + c1 * aj * dphi_0 || phij >= phi_lo) {
                ahi = aj;
                phi_hi = phij;
            } else {
                // Check Curvature
                if (std::abs(dphij) <= -c2 * dphi_0) {
                    C_new = Cj; X_new = Xj;
                    return aj; // Optimal found
                }
                if (dphij * (ahi - alo) >= 0) {
                    ahi = alo;
                    phi_hi = phi_lo;
                }
                alo = aj;
                phi_lo = phij;
                dphi_lo = dphij;
            }
        }
        // Fallback to lo
        // Need to set C_new/X_new to alo state
        double check_phi, check_dphi;
        eval_step(alo, check_phi, check_dphi, X_new, C_new);
        return alo;
    };

    // Main Line Search Loop (Algorithm 3.5)
    for(int i=0; i<10; ++i) {
        arma::mat Xi, Ci;
        double phi_curr, dphi_curr;
        eval_step(alpha_curr, phi_curr, dphi_curr, Xi, Ci);
        
        if ( (phi_curr > phi_0 + c1 * alpha_curr * dphi_0) || (i > 0 && phi_curr >= phi_prev) ) {
            // Need dphi at alpha_prev which is dphi_0 for i=0.
            double dphi_prev_iter = (i==0) ? dphi_0 : dphi_prev;
            return zoom(alpha_prev, alpha_curr, phi_prev, phi_curr, dphi_prev_iter); 
        }
        
        if (std::abs(dphi_curr) <= -c2 * dphi_0) {
            C_new = Ci; X_new = Xi;
            return alpha_curr; // Strong Wolfe satisfied
        }
        
        if (dphi_curr >= 0) {
            return zoom(alpha_curr, alpha_prev, phi_curr, phi_prev, dphi_curr);
        }
        
        alpha_prev = alpha_curr;
        phi_prev = phi_curr;
        dphi_prev = dphi_curr;
        
        alpha_curr *= 1.5; // Simple extrapolation
        if(alpha_curr > alpha_max) alpha_curr = alpha_max;
    }
    
    return 0.0; // Failed
}

} // namespace rdmft
} // namespace helfem
