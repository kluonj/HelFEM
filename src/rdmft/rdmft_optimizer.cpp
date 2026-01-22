#include "rdmft_optimizer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace helfem {
namespace rdmft {

// =============================================================================
// Helper: Strong Wolfe Line Search
// =============================================================================

// Helper for zoom phase
static double zoom(Optimizer::State& current_state,
                   const arma::mat& dir_C, const arma::vec& dir_n,
                   Optimizer::EnergyGradFunc energy_func,
                   Optimizer::ManifoldGradFunc manifold_grad_func,
                   Optimizer::RetractFunc retract_func,
                   Optimizer::TransportFunc transport_func,
                   Optimizer::InnerProductFunc inner_product_func,
                   double dphi_0, double phi_0,
                   double alpha_lo, double alpha_hi,
                   double phi_lo, double phi_hi,
                   double dphi_lo) {
    double c1 = 1e-4;
    double c2 = 0.9;
    int max_zoom = 20;

    for (int i = 0; i < max_zoom; ++i) {
        // Interpolate (Quadratic or Cubic) - simplistic bisection/quadratic here for stability
        // Use quadratic interpolation between lo and hi using f(lo), f'(lo), f(hi)
        // phi(alpha) = a*alpha^2 + b*alpha + c relative to lo
        // better: just cubic interpolation
        
        double d1 = phi_lo - phi_hi; // >?
        double d2 = dphi_lo;
        // Bisection for now as robust fallback
        double alpha_j = 0.5 * (alpha_lo + alpha_hi);
        
        // Quadratic interp if possible
        if (i > 0) {
           // Fit quadratic to phi_lo, phi_hi, dphi_lo?
           // Or just trust bisection for robustness
        }
        
        Optimizer::State trial = current_state;
        // Retract
        if (retract_func) {
            retract_func(current_state, dir_C, dir_n, alpha_j, trial);
        } else {
            if(dir_C.n_elem > 0) trial.C = current_state.C + alpha_j * dir_C;
            if(dir_n.n_elem > 0) trial.n = current_state.n + alpha_j * dir_n;
        }
        
        // Eval
        arma::mat gC_dummy; arma::vec gn_dummy;
        trial.energy = energy_func(trial.C, trial.n, gC_dummy, gn_dummy);
        double phi_j = trial.energy;
        
        // Armijo check part 1
        // We use dphi_0 for sufficient decrease check
        double armijo_term = c1 * alpha_j * dphi_0;
        
        if ((phi_j > phi_0 + armijo_term) || (phi_j >= phi_lo)) {
            alpha_hi = alpha_j;
            phi_hi = phi_j;
        } else {
            // Need gradient at alpha_j to check curvature
            // Update trial gradients
             trial.gC = gC_dummy;
             trial.gn = gn_dummy;
             // Project if needed
             if (manifold_grad_func) {
                manifold_grad_func(trial.C, trial.n, trial.gC, trial.gn);
             }
             
             // Compute dphi_j = <grad_j, transported_dir>
             double dphi_j = 0.0;
             if (transport_func) {
                 arma::mat trans_dir_C = dir_C; arma::vec trans_dir_n = dir_n;
                 transport_func(current_state, trial, trans_dir_C, trans_dir_n);
                 if (inner_product_func) {
                     dphi_j = inner_product_func(trial, trial.gC, trial.gn, trans_dir_C, trans_dir_n);
                 } else {
                     if(dir_C.n_elem > 0) dphi_j += arma::dot(trial.gC, trans_dir_C);
                     if(dir_n.n_elem > 0) dphi_j += arma::dot(trial.gn, trans_dir_n);
                 }
             } else {
                 // Euclidean fallback - no transport
                  if(dir_C.n_elem > 0) dphi_j += arma::dot(trial.gC, dir_C);
                  if(dir_n.n_elem > 0) dphi_j += arma::dot(trial.gn, dir_n);
             }

             // Curvature check
             if (std::abs(dphi_j) <= -c2 * dphi_0) {
                 current_state = trial;
                 return alpha_j;
             }
             
             if (dphi_j * (alpha_hi - alpha_lo) >= 0) {
                 alpha_hi = alpha_lo;
                 phi_hi = phi_lo;
             }
             alpha_lo = alpha_j;
             phi_lo = phi_j;
             dphi_lo = dphi_j;
        }
    }
    return alpha_lo; // Fallback
}

static double strong_wolfe_linesearch(Optimizer::State& current_state, 
                                      const arma::mat& dir_C, const arma::vec& dir_n,
                                      Optimizer::EnergyGradFunc energy_func, 
                                      Optimizer::ManifoldGradFunc manifold_grad_func,
                                      Optimizer::RetractFunc retract_func,
                                      Optimizer::TransportFunc transport_func,
                                      Optimizer::InnerProductFunc inner_product_func,
                                      double dphi_0,
                                      double initial_step = 1.0) {
    double alpha = initial_step;
    double alpha_max = 10.0;
    double c1 = 1e-4;
    double c2 = 0.9;
    
    double alpha_prev = 0.0;
    double phi_0 = current_state.energy;
    double phi_prev = phi_0;
    double dphi_prev = dphi_0;
    
    int max_ls = 20;

    for (int i = 1; i <= max_ls; ++i) {
        // Trial State
        Optimizer::State trial = current_state;
        if (retract_func) {
            retract_func(current_state, dir_C, dir_n, alpha, trial);
        } else {
            if(dir_C.n_elem > 0) trial.C = current_state.C + alpha * dir_C;
            if(dir_n.n_elem > 0) trial.n = current_state.n + alpha * dir_n;
        }
        
        // Eval
        arma::mat gC_dummy; arma::vec gn_dummy;
        trial.energy = energy_func(trial.C, trial.n, gC_dummy, gn_dummy);
        double phi_curr = trial.energy;
        
        // Armijo check
        if ((phi_curr > phi_0 + c1 * alpha * dphi_0) || (i > 1 && phi_curr >= phi_prev)) {
             double res = zoom(current_state, dir_C, dir_n, energy_func, manifold_grad_func, retract_func, transport_func, inner_product_func, dphi_0, phi_0, alpha_prev, alpha, phi_prev, phi_curr, dphi_prev);
             return res > 1e-10 ? res : 0.0;
        }
        
        // Gradient at trial
        trial.gC = gC_dummy; trial.gn = gn_dummy;
        if (manifold_grad_func) manifold_grad_func(trial.C, trial.n, trial.gC, trial.gn);
        
        double dphi_curr = 0.0;
         if (transport_func) {
             arma::mat trans_dir_C = dir_C; arma::vec trans_dir_n = dir_n;
             transport_func(current_state, trial, trans_dir_C, trans_dir_n);
             if (inner_product_func) {
                 dphi_curr = inner_product_func(trial, trial.gC, trial.gn, trans_dir_C, trans_dir_n);
             } else {
                 if(dir_C.n_elem > 0) dphi_curr += arma::dot(trial.gC, trans_dir_C);
                 if(dir_n.n_elem > 0) dphi_curr += arma::dot(trial.gn, trans_dir_n);
             }
         } else {
              if(dir_C.n_elem > 0) dphi_curr += arma::dot(trial.gC, dir_C);
              if(dir_n.n_elem > 0) dphi_curr += arma::dot(trial.gn, dir_n);
         }
         
         // Curvature check
         if (std::abs(dphi_curr) <= -c2 * dphi_0) {
             current_state = trial;
             return alpha;
         }
         
         if (dphi_curr >= 0) {
             double res = zoom(current_state, dir_C, dir_n, energy_func, manifold_grad_func, retract_func, transport_func, inner_product_func, dphi_0, phi_0, alpha, alpha_prev, phi_curr, phi_prev, dphi_curr);
             return res > 1e-10 ? res : 0.0;
         }
         
         alpha_prev = alpha;
         phi_prev = phi_curr;
         dphi_prev = dphi_curr;
         
         alpha = std::min(alpha * 2.0, alpha_max);
    }
    
    return alpha;
}

// =============================================================================
// Helper: Simple Backtracking Line Search
// =============================================================================
static double backtracking_linesearch(Optimizer::State& current_state, 
                                      const arma::mat& dir_C, const arma::vec& dir_n,
                                      Optimizer::EnergyGradFunc energy_func, 
                                      Optimizer::RetractFunc retract_func,
                                      double dphi_0,
                                      double initial_step = 1.0) {
    double alpha = initial_step;
    double rho = 0.5;
    double c1 = 1e-4;
    int max_ls = 50;
    
    Optimizer::State trial = current_state; // Copy metadata (energy, etc.)
    
    for(int i=0; i<max_ls; ++i) {
        // Retract
        if (retract_func) {
            retract_func(current_state, dir_C, dir_n, alpha, trial);
        } else {
            // Default Euclidean step
            if(dir_C.n_elem > 0) trial.C = current_state.C + alpha * dir_C;
            if(dir_n.n_elem > 0) trial.n = current_state.n + alpha * dir_n;
        }
        
        // Eval Energy (updates gC, gn in trial to Euclidean Gradients usually, 
        // but we only check energy here)
        // Note: We might optimize by NOT computing gradients during line search if possible,
        // but the interface requires it.
        arma::mat gC_dummy; arma::vec gn_dummy;
        trial.energy = energy_func(trial.C, trial.n, gC_dummy, gn_dummy);
        
        // Armijo condition
        double sufficient_decrease_term;
        if (retract_func) {
             // For Projected Gradient or Manifold Retraction, use effective step
             double d_dot_step = 0.0;
             if(current_state.C.n_elem > 0) d_dot_step += arma::dot(current_state.gC, trial.C - current_state.C);
             if(current_state.n.n_elem > 0) d_dot_step += arma::dot(current_state.gn, trial.n - current_state.n);
             sufficient_decrease_term = c1 * d_dot_step;
        } else {
             sufficient_decrease_term = c1 * alpha * dphi_0;
        }

        if (trial.energy <= current_state.energy + sufficient_decrease_term) {
            current_state = trial;
            // Now we should update gradients properly using the finalized trial state
            // Re-eval with gradients if needed (EnergyGradFunc computed them into dummies)
            // But we actually need the gradients for the NEXT optimization step.
            // So we update the state's gradients.
            current_state.gC = gC_dummy;
            current_state.gn = gn_dummy;
            return alpha;
        }
        
        alpha *= rho;
    }
    return 0.0;
}

// =============================================================================
// Steepest Descent
// =============================================================================
void SteepestDescentOptimizer::optimize(State& state, 
                                        EnergyGradFunc energy_func, 
                                        ManifoldGradFunc manifold_grad_func,
                                        RetractFunc retract_func,
                                        TransportFunc transport_func,
                                        InnerProductFunc inner_product_func) {
    if (verbose_) std::cout << "  [SD] Starting Steepest Descent Optimization\n";

    // Initial eval
    state.energy = energy_func(state.C, state.n, state.gC, state.gn);
    
    // Store previous state for BB/Quad steps
    State old_state = state;
    arma::mat old_gC = state.gC;
    arma::vec old_gn = state.gn;
    // We also need gradients projected!
    if (manifold_grad_func) {
        manifold_grad_func(state.C, state.n, old_gC, old_gn); // Compute initial grad
        state.gC = old_gC; state.gn = old_gn; // Assign back
    }
    double prev_step_size = 0.0;
    double E_very_prev = state.energy; // E_{k-1}

    for (int iter = 0; iter < max_iter_; ++iter) {
        double E_prev = state.energy; // E_k before update
        
        // Project Euclidean gradient to Riemannian gradient if needed
        // (If not done at end of previous step)
        if (manifold_grad_func) {
            manifold_grad_func(state.C, state.n, state.gC, state.gn);
        }
        
        // Determine search direction (negative gradient)
        arma::mat dir_C = -state.gC;
        arma::vec dir_n = -state.gn; 
        
        // Compute directional derivative (Inner Product)
        double dphi_0 = 0.0;
        if (inner_product_func) {
             dphi_0 = inner_product_func(state, state.gC, state.gn, dir_C, dir_n);
        } else {
             // Default Euclidean dot product
             if(state.C.n_elem > 0) dphi_0 += arma::dot(state.gC, dir_C);
             if(state.n.n_elem > 0) dphi_0 += arma::dot(state.gn, dir_n);
        }
        
        // Make sure it is descent
        if (dphi_0 > 0) {
             // Should not happen for SD unless numerical noise
             dir_C = -dir_C;
             dir_n = -dir_n;
             dphi_0 = -dphi_0;
        }

        // --- Determine Initial Step ---
        double initial_step = 1.0;
        double g_norm_sq = std::abs(dphi_0);
        double g_norm = std::sqrt(g_norm_sq);
        
        if (iter == 0) {
            if (init_step_type_ == InitialStep::Fixed) initial_step = 1.0;
            else if (g_norm > 1.0) initial_step = 1.0 / g_norm;
            else initial_step = 1.0;
        } else {
            switch(init_step_type_) {
                case InitialStep::Fixed:
                    initial_step = 1.0; 
                    break;
                case InitialStep::InverseNorm:
                    // Usually 1/|g| is too small for later steps, but maybe user wants it
                    if(g_norm > 1e-12) initial_step = 1.0 / g_norm;
                    break;
                case InitialStep::Quadratic:
                {
                    // Quad interp: alpha = 2 * (E_prev - E_curr) / |g_curr|^2?
                    // No, that assumes minimal at alpha.
                    // Or alpha = 2 * (E_{k-1} - E_k) / |g_k|^2  <-- Nocedal Wright?
                    // Actually, often: alpha_0 = alpha_{k-1} * |g_{k-1}^T d_{k-1}| / |g_k^T d_k| ?
                    // Let's use the interpolation based on previous step size and energy change.
                    // alpha = 2 * (E_{k-1} - E_k) / |g_{k-1}|^2 ?? No.
                    // Nocedal Eq 3.60: alpha_0 = 2*(E_{k} - E_{k-1}) / phi'(0)
                    // Here E_{k} is current function value, E_{k-1} is PREVIOUS.
                    // phi'(0) is dphi_0 (negative).
                    // alpha = 2 * (E_curr - E_prev_step) / dphi_0
                    // Since E_curr < E_prev_step (descent), numerator is negative. dphi_0 is negative.
                    // So alpha is positive.
                    double E_curr = state.energy;
                    double E_last = E_very_prev; // E from start of PREVIOUS iteration
                    // Wait, E_prev variable here is actually E_curr (start of THIS iter).
                    // We need E from start of LAST iter.
                    // Let's rely on stored check.
                    double energy_change = E_curr - E_last; // negative
                    if (std::abs(dphi_0) > 1e-14) {
                        initial_step = 2.0 * energy_change / dphi_0;
                    }
                    initial_step = std::min(1.1 * prev_step_size, initial_step); // Limit growth
                }
                break;
                case InitialStep::BarzilaiBorwein:
                {
                     // s = x_k - x_{k-1}
                     // y = g_k - g_{k-1}
                     // But on manifold?
                     // BB roughly: alpha = <s,s>/<s,y>
                     // Transport s and y?
                     // s is effectively step * dir_{k-1} (retracted).
                     // Ideally we use vector transport for y.
                     
                     // Reconstruct s from previous step info
                     // Current State is x_k. old_state is x_{k-1}.
                     // Actually, we need to carefully track old_state at the TOP of the loop.
                     // I added `State old_state = state` at start of loop, but that gets overwritten?
                     // No, I added it before loop. Inside loop we need to update it.
                     
                     // We need y_{k-1} in T_k.
                     // y = g_k - Transport(g_{k-1})
                     // s = Transport(s_{k-1}) approx step * Transport(dir_{k-1})
                     
                     // Let's try simple Euclidean BB if no transport func, or Riemannian if available.
                     // The BFGS implementation already computes s and y!
                     // But this is Steepest Descent.
                     
                     arma::mat s_C; arma::vec s_n;
                     // s = x_new - x_old ?? No, s is the step vector in T_{k-1} usually.
                     // s_{k-1} = step_{k-1} * dir_{k-1}.
                     // But we are at k. We need <s, s> / <s, y>.
                     // BB1: <s_{k-1}, s_{k-1}> / <s_{k-1}, y_{k-1}>. All in T_{k-1}.
                     
                     // We have:
                     // old_gC, old_gn (Gradient at k-1)
                     // old_state (State at k-1)
                     // but we don't have dir_{k-1} stored easily unless we keep it.
                     // But we can approximate s ~ step_{k-1} * dir_{k-1}.
                     // y = g_prev_step_at_new - g_prev. 
                     // Wait, y needs gradient at NEW point transported to OLD point or vice versa.
                     // Usually BB uses: s_{k-1} and y_{k-1}.
                     // y_{k-1} = g_k - g_{k-1}.
                     // However g_k is in T_k, g_{k-1} is in T_{k-1}.
                     // We need to transport g_{k-1} to T_k to subtract? Or g_k to T_{k-1}?
                     // Standard is: s_{k-1} in T_{k-1}, y_{k-1} in T_{k-1}.
                     // y_{k-1} = Transport^{-1}(g_k) - g_{k-1}. (Pullback)
                     // Actually, usually transport OLD to NEW, then compute in NEW tangent space.
                     // s_{k-1} -> Transp(s_{k-1}). y_{k-1} = g_k - Transp(g_{k-1}).
                     
                     // We need to have saved g_{k-1} (old_gC, old_gn).
                     // And s_{k-1}. We didn't save s_{k-1}.
                     // Let's save `last_step * last_dir` at end of loop?
                     // Or just approximate.
                     
                     // To properly implement BB, we need to change the loop structure to store history.
                     // Let's do a simplified approach:
                     // If we are at iter > 0, we assume we have stored `last_s_C`, `last_s_n` (transported?)
                     // This is getting complicated for SD without a stored history class.
                     
                     // Alternative: Euclidean Approx for BB
                     // s = C_curr - C_old. 
                     // y = gC_curr - gC_old.
                     // alpha = dot(s,s)/dot(s,y).
                     
                     if (prev_step_size > 0 && iter > 0) {
                         // Approx s
                         arma::mat sC_diff = state.C - old_state.C;
                         arma::vec sn_diff = state.n - old_state.n;
                         
                         arma::mat yC_diff = state.gC - old_gC;
                         arma::vec yn_diff = state.gn - old_gn;
                         
                         double s_s = 0;
                         if(state.C.n_elem>0) s_s += arma::dot(sC_diff, sC_diff);
                         if(state.n.n_elem>0) s_s += arma::dot(sn_diff, sn_diff);
                         
                         double s_y = 0;
                         if(state.C.n_elem>0) s_y += arma::dot(sC_diff, yC_diff);
                         if(state.n.n_elem>0) s_y += arma::dot(sn_diff, yn_diff);
                         
                         if(std::abs(s_y) > 1e-14) {
                             initial_step = s_s / s_y;
                         } else {
                             initial_step = prev_step_size;
                         }
                         
                         if (initial_step <= 0) initial_step = prev_step_size; // Fallback if negative curvature locally
                     }
                }
                break;
            }
        }
        
        // Safety clamp
        if (initial_step < 1e-8) initial_step = 1e-8;
        if (initial_step > 10.0) initial_step = 10.0;
        
        // Store for next iter
        old_state = state; // Deep copy
        old_gC = state.gC; old_gn = state.gn;
        E_very_prev = E_prev; // The energy at START of this iter
                            // (which becomes "Energy at start of Prev Iter" in next loop)

        double step = backtracking_linesearch(state, dir_C, dir_n, energy_func, retract_func, dphi_0, initial_step);
        
        prev_step_size = step;

        
        double dE = std::abs(state.energy - E_prev);
        if (verbose_) {
             std::cout <<  std::setprecision(8) << "  [SD] Iter " << iter << " E=" << state.energy << " dE=" << dE << " Step=" << step << "\n";
        }

        if (step == 0.0) {
            if (verbose_) std::cout << "  [SD] Line search failed.\n";
            break;
        }
        
        if (dE < energy_tol_ || step < step_tol_) break;
    }
}

// =============================================================================
// Conjugate Gradient (Polak-Ribiere)
// =============================================================================
void ConjugateGradientOptimizer::optimize(State& state, 
                                          EnergyGradFunc energy_func, 
                                          ManifoldGradFunc manifold_grad_func,
                                          RetractFunc retract_func,
                                          TransportFunc transport_func,
                                          InnerProductFunc inner_product_func) {
    if (verbose_) std::cout << "  [CG] Starting Conjugate Gradient Optimization\n";

    // Initial eval
    state.energy = energy_func(state.C, state.n, state.gC, state.gn);
    
    // Initial Manifold Grad
    if (manifold_grad_func) manifold_grad_func(state.C, state.n, state.gC, state.gn);

    // Initial Search Direction
    arma::mat dir_C = -state.gC;
    arma::vec dir_n = -state.gn;

    // Previous iteration storage
    arma::mat prev_gC = state.gC;
    arma::vec prev_gn = state.gn;
    arma::mat prev_dir_C = dir_C;
    arma::vec prev_dir_n = dir_n;

    for (int iter = 0; iter < max_iter_; ++iter) {
        double E_prev = state.energy;
        
        // Directional derivative dphi_0 = <grad, dir>
        double dphi_0 = 0.0;
        if (inner_product_func) {
             dphi_0 = inner_product_func(state, state.gC, state.gn, dir_C, dir_n);
        } else {
             if(state.C.n_elem > 0) dphi_0 += arma::dot(state.gC, dir_C);
             if(state.n.n_elem > 0) dphi_0 += arma::dot(state.gn, dir_n);
        }

        // Restart if not descent
        if (dphi_0 >= 0) {
            if (verbose_) std::cout << "  [CG] Restart (non-descent)\n";
            dir_C = -state.gC;
            dir_n = -state.gn;
            if (inner_product_func) {
                dphi_0 = inner_product_func(state, state.gC, state.gn, dir_C, dir_n);
            } else {
                dphi_0 = 0.0;
                if(state.C.n_elem > 0) dphi_0 += arma::dot(state.gC, dir_C);
                if(state.n.n_elem > 0) dphi_0 += arma::dot(state.gn, dir_n);
            }
        }
        
        // Save current gradients and direction before update (for Polak Ribiere and Transport)
        prev_gC = state.gC;
        prev_gn = state.gn;
        prev_dir_C = dir_C;
        prev_dir_n = dir_n;
        State prev_state = state; // Deep copy
        
        // Line Search
        // Note: backtracking_linesearch updates 'state' to the new point
        // and also updates state.gC/gn to the Euclidean gradient at the new point.
        double step = backtracking_linesearch(state, dir_C, dir_n, energy_func, retract_func, dphi_0);
        
        double dE = std::abs(state.energy - E_prev);
        if (verbose_) {
             std::cout <<  std::setprecision(8) << "  [CG] Iter " << iter << " E=" << state.energy << " dE=" << dE << " Step=" << step << "\n";
        }
        
        if (step == 0.0) {
             if (verbose_) std::cout << "  [CG] Line search failed (step too small).\n";
             break;
        }
        if (dE < energy_tol_ || step < step_tol_) break;

        // At new state:
        // 1. Calculate Riemannian Gradient at new state
        if (manifold_grad_func) manifold_grad_func(state.C, state.n, state.gC, state.gn);
        // Now state.gC is G_{new}

        // 2. Vector Transport prev_gC and prev_dir_C to new state
        arma::mat trans_prev_gC = prev_gC;
        arma::vec trans_prev_gn = prev_gn;
        arma::mat trans_prev_dir_C = prev_dir_C; 
        arma::vec trans_prev_dir_n = prev_dir_n;

        if (transport_func) {
             transport_func(prev_state, state, trans_prev_gC, trans_prev_gn);
             transport_func(prev_state, state, trans_prev_dir_C, trans_prev_dir_n);
        }
        
        // 3. Polak-Ribiere Beta
        // beta = <G_new, G_new - G_old_trans> / <G_old, G_old>
        // Note: Denominator uses G_old at Old point (so no transport needed for denominator if norm is preserved or using inner product at old point)
        // Actually, typically <G_old, G_old>_old.
        
        double num = 0.0;
        double den = 0.0;
        
        if (inner_product_func) {
            // Difference vector at new point
            arma::mat diff_gC = state.gC - trans_prev_gC;
            arma::vec diff_gn = state.gn - trans_prev_gn;
            
            num = inner_product_func(state, state.gC, state.gn, diff_gC, diff_gn);
            den = inner_product_func(prev_state, prev_gC, prev_gn, prev_gC, prev_gn); 
        } else {
            // Euclidean Fallback
            if (state.C.n_elem > 0) {
                num += arma::dot(state.gC, state.gC - trans_prev_gC);
                den += arma::dot(prev_gC, prev_gC);
            }
            if (state.n.n_elem > 0) {
                num += arma::dot(state.gn, state.gn - trans_prev_gn);
                den += arma::dot(prev_gn, prev_gn);
            }
        }
        
        double beta = 0.0;
        if (std::abs(den) > 1e-14) beta = std::max(0.0, num / den);
        
        // 4. Update Direction
        // d_new = -G_new + beta * d_old_trans
        if (state.C.n_elem > 0) dir_C = -state.gC + beta * trans_prev_dir_C;
        if (state.n.n_elem > 0) dir_n = -state.gn + beta * trans_prev_dir_n;
    }
}

// =============================================================================
// BFGS (Riemannian Limited-memory BFGS)
// =============================================================================
void BFGSOptimizer::optimize(State& state, 
                             EnergyGradFunc energy_func, 
                             ManifoldGradFunc manifold_grad_func,
                             RetractFunc retract_func,
                             TransportFunc transport_func,
                             InnerProductFunc inner_product_func) {
    if (verbose_) std::cout << "  [BFGS] Starting L-BFGS Optimization\n";

    // Configuration
    int m = 5; // Memory size
    
    // Initial eval
    state.energy = energy_func(state.C, state.n, state.gC, state.gn);
    if (manifold_grad_func) {
        manifold_grad_func(state.C, state.n, state.gC, state.gn);
    }

    // Local Helper: Dot Product
    auto dot_prod = [&](const arma::mat& aC, const arma::vec& an, const arma::mat& bC, const arma::vec& bn) -> double {
        if (inner_product_func) return inner_product_func(state, aC, an, bC, bn);
        double val = 0.0;
        if(aC.n_elem > 0 && bC.n_elem > 0) val += arma::dot(aC, bC);
        if(an.n_elem > 0 && bn.n_elem > 0) val += arma::dot(an, bn);
        return val;
    };

    struct Pair {
        arma::mat s_C; arma::vec s_n;
        arma::mat y_C; arma::vec y_n;
        double rho;
    };
    std::deque<Pair> history;

    for (int iter = 0; iter < max_iter_; ++iter) {
        double E_prev = state.energy;
        double g_norm = std::sqrt(dot_prod(state.gC, state.gn, state.gC, state.gn));

        if (g_norm < grad_tol_) {
             if (verbose_) std::cout << "  [BFGS] Converged |g|=" << g_norm << "\n";
             break;
        }

        // --- L-BFGS Two-Loop Recursion ---
        // q = g
        arma::mat q_C = state.gC;
        arma::vec q_n = state.gn;
        
        std::vector<double> alphas(history.size());
        
        // Backward pass
        for (int i = (int)history.size() - 1; i >= 0; --i) {
            double alpha = history[i].rho * dot_prod(history[i].s_C, history[i].s_n, q_C, q_n);
            alphas[i] = alpha;
            // q = q - alpha * y
            if(q_C.n_elem > 0) q_C -= alpha * history[i].y_C;
            if(q_n.n_elem > 0) q_n -= alpha * history[i].y_n;
        }

        // Scaling gamma
        double gamma = 1.0;
        if (!history.empty()) {
            const auto& last = history.back();
            double y_dot_y = dot_prod(last.y_C, last.y_n, last.y_C, last.y_n);
            double s_dot_y = dot_prod(last.s_C, last.s_n, last.y_C, last.y_n); // actually 1/rho
            if (y_dot_y > 1e-14) gamma = s_dot_y / y_dot_y;
        }
        
        // r = gamma * q
        arma::mat r_C = gamma * q_C;
        arma::vec r_n = gamma * q_n;

        // Forward pass
        for (int i = 0; i < (int)history.size(); ++i) {
            double beta = history[i].rho * dot_prod(history[i].y_C, history[i].y_n, r_C, r_n);
            double coeff = alphas[i] - beta;
            // r = r + coeff * s
            if(r_C.n_elem > 0) r_C += coeff * history[i].s_C;
            if(r_n.n_elem > 0) r_n += coeff * history[i].s_n;
        }

        // dir = -r
        arma::mat dir_C = -r_C;
        arma::vec dir_n = -r_n;

        // Check descent direction
        double dphi_0 = dot_prod(state.gC, state.gn, dir_C, dir_n);
        if (dphi_0 > 0) {
            // Fallback to SD if not descent (can happen with quasi-newton)
            if (verbose_) std::cout << "  [BFGS] Non-descent (" << dphi_0 << "), resetting history.\n";
            history.clear();
            dir_C = -state.gC;
            dir_n = -state.gn;
            dphi_0 = dot_prod(state.gC, state.gn, dir_C, dir_n);
        }

        // Store old state / gradient for y computation
        State old_state = state; // copies C, n, gC, gn
        
        // Line Search
        // Use proper Strong Wolfe Line Search if possible, falling back to backtracking only if really needed?
        
        // Initial Step Size Choice
        double initial_step = 1.0;
        
        // Custom strategy overrides
        if (iter == 0) {
            // First step heuristic based on type or norm
            if (init_step_type_ == InitialStep::Fixed) initial_step = 1.0;
            else if (g_norm > 1.0) initial_step = 1.0 / g_norm;
            else initial_step = 1.0;
        } else {
            // iter > 0
            switch(init_step_type_) {
                case InitialStep::Fixed: 
                case InitialStep::InverseNorm: // For BFGS, usually stick to 1.0 after start
                    initial_step = 1.0; 
                    break;
                case InitialStep::BarzilaiBorwein:
                    if (!history.empty()) {
                         // Use latest pair
                         const auto& last = history.back();
                         double y_dot_y = dot_prod(last.y_C, last.y_n, last.y_C, last.y_n);
                         double s_dot_s = dot_prod(last.s_C, last.s_n, last.s_C, last.s_n);
                         double s_dot_y = dot_prod(last.s_C, last.s_n, last.y_C, last.y_n);
                         
                         // BB1: <s,s>/<s,y>
                         // BB2: <s,y>/<y,y>
                         // Use BB1 often
                         if (std::abs(s_dot_y) > 1e-14) {
                             initial_step = s_dot_s / s_dot_y; 
                         }
                    } else {
                        initial_step = 1.0;
                    }
                    break;
                case InitialStep::Quadratic:
                    double energy_change = state.energy - E_prev; // Negative
                    if (std::abs(dphi_0) > 1e-14) {
                        initial_step = 2.0 * std::abs(energy_change) / std::abs(dphi_0);
                    }
                    if (initial_step <= 0) initial_step = 1.0;
                    break;
            }
        }
        
        // Safety clamp for BFGS to avoid destroying superlinear conv if it gets crazy
        if (initial_step < 1e-4) initial_step = 1e-4; // Don't start too small
        if (initial_step > 10.0) initial_step = 10.0; // Don't start too big
        
        
        double step;
        bool use_strong_wolfe = true; 
        
        if (use_strong_wolfe) {
             step = strong_wolfe_linesearch(state, dir_C, dir_n, energy_func, manifold_grad_func, retract_func, transport_func, inner_product_func, dphi_0, initial_step);
        } else {
             step = backtracking_linesearch(state, dir_C, dir_n, energy_func, retract_func, dphi_0, initial_step);
        }

        if (step == 0.0) {
             if (verbose_) std::cout << "  [BFGS] Line search failed (step too small).\n";
             break;
        }

        double dE = std::abs(state.energy - E_prev);
        if (verbose_) {
             std::cout <<  std::setprecision(8) << "  [BFGS] Iter " << iter << " E=" << state.energy << " dE=" << dE << " Step=" << step << "\n";
        }
        
        if (dE < energy_tol_ || dE < 1e-13) { // Very small change
             // break; // Let outer loops handle convergence?
        }
        
        // Note: strong_wolfe linesearch has updated state.gC to Euclidean gradient at new point.
        // It has NOT applied manifold_grad_func yet (wait, yes it did internally but then might have moved again? No, it returns last trial).
        // Actually zoom calls manifold_grad_func on trial.
        
        // Project new gradient (just in case linesearch didn't do it perfectly or implementation variance)
        if (manifold_grad_func) {
            manifold_grad_func(state.C, state.n, state.gC, state.gn);
        }

        // --- Compute s_k and y_k ---
        // s_k on Manifold:
        // Generally s_k = Transp^{-1}(x_new, x_old, step*dir) ... wait.
        // Actually, vector transport moves from old -> new.
        // s_k is the vector in T_{x_k} that maps to x_{k+1}. 
        // For retraction R_x(eta), x_{k+1} = R_x(step * dir). So s_k approx step * dir.
        // But BFGS usually wants s and y in the SAME tangent space to take dot products?
        // No, standard Riemannian BFGS stores pairs (s,y) where s \in T_x, y \in T_x.
        // But during the loop, we are at x_k, and we look back at x_{k-1}.
        // We need to transport vectors from x_{k-1} to x_k.
        
        // s_{k-1} was in T_{k-1}. y_{k-1} was in T_{k-1}.
        // We transport them to T_k.
        
        // First, define s_new and y_new in T_{old}.
        // s_new = step * dir. (Already in T_{old})
        arma::mat s_C_new = step * dir_C;
        arma::vec s_n_new = step * dir_n;

        // y_new = grad_new (transported to old) - grad_old
        // Wait, standard definition: y_k = grad_{k+1} - grad_k?
        // No, y_k in T_{k}. y_k = grad_{k+1} - Transp(k->k+1, grad_k)? No.
        // Usually y_k approx Hess * s_k.
        // If s_k is in T_k, then y_k should be in T_k.
        // y_k = Transp^{-1}(new->old, grad_new) - grad_old.
        // OR define everything in T_{new}.
        // s_k = Transp(old->new, step*dir).
        // y_k = grad_new - Transp(old->new, grad_old).
        
        // Let's use the T_{new} convention (Huang 2015).
        // Transport OLD gradient to NEW space.
        arma::mat g_old_trans_C = old_state.gC;
        arma::vec g_old_trans_n = old_state.gn;
        if (transport_func) {
            transport_func(old_state, state, g_old_trans_C, g_old_trans_n);
        }

        // Transport step s=step*dir to NEW space efficiently
        // s_raw = step * dir (in Old space)
        arma::mat s_trans_C = s_C_new;
        arma::vec s_trans_n = s_n_new; 
        if (transport_func) {
            transport_func(old_state, state, s_trans_C, s_trans_n);
        }
        
        Pair new_pair;
        new_pair.s_C = s_trans_C;
        new_pair.s_n = s_trans_n;
        new_pair.y_C = state.gC - g_old_trans_C;
        new_pair.y_n = state.gn - g_old_trans_n;
        
        double s_dot_y = dot_prod(new_pair.s_C, new_pair.s_n, new_pair.y_C, new_pair.y_n);
        
        if (s_dot_y > 1e-14) {
             new_pair.rho = 1.0 / s_dot_y;
             
             // Transport History
             if (transport_func) {
                 for(auto& p : history) {
                     transport_func(old_state, state, p.s_C, p.s_n);
                     transport_func(old_state, state, p.y_C, p.y_n);
                 }
             }
             
             history.push_back(new_pair);
             if((int)history.size() > m) history.pop_front();
        } else {
             if(verbose_) std::cout << "  [BFGS] curvature condition failed s.y=" << s_dot_y << "\n";
        }
    }
}
} // namespace rdmft
} // namespace helfem
