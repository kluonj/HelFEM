#include "optimizer.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace helfem {
namespace rdmft {

namespace {
inline double mat_dot(const arma::mat& A, const arma::mat& B) {
    return arma::dot(A, B);
}

inline arma::mat symm(const arma::mat& A) {
    return 0.5 * (A + A.t());
}

inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

double cubic_interpolate(double x1, double f1, double g1,
                         double x2, double f2, double g2,
                         double x_min, double x_max) {
    double denom = (x1 - x2);
    if (std::abs(denom) < 1e-14) return 0.5 * (x_min + x_max);

    double d1 = g1 + g2 - 3.0 * (f1 - f2) / denom;
    double disc = d1 * d1 - g1 * g2;
    if (disc < 0.0) return 0.5 * (x_min + x_max);

    double d2 = std::sqrt(disc);
    if (x2 < x1) d2 = -d2;

    double num = g2 + d2 - d1;
    double den = g2 - g1 + 2.0 * d2;
    if (std::abs(den) < 1e-14) return 0.5 * (x_min + x_max);

    double x = x2 - (x2 - x1) * (num / den);
    if (!std::isfinite(x)) return 0.5 * (x_min + x_max);
    return clamp(x, x_min, x_max);
}

} // namespace

struct OrbitalOptimizer::EvalResult {
    double E = 0.0;
    arma::mat X;
    arma::mat C;
    arma::mat grad;
};

arma::mat OrbitalOptimizer::to_orthogonal_basis(const arma::mat& C, const arma::mat& S_sqrt) const {
    return S_sqrt.t() * C;
}

arma::mat OrbitalOptimizer::from_orthogonal_basis(const arma::mat& X, const arma::mat& S_inv_sqrt) const {
    return S_inv_sqrt * X;
}

arma::mat OrbitalOptimizer::project_to_tangent(const arma::mat& X, const arma::mat& V) const {
    return V - X * symm(X.t() * V);
}

arma::mat OrbitalOptimizer::calc_riem_grad(const arma::mat& X, const arma::mat& gX) const {
    return project_to_tangent(X, gX);
}

arma::mat OrbitalOptimizer::retract(const arma::mat& X_trial) const {
    arma::mat X_out = X_trial;

    if (retraction_ == Retraction::QR) {
             arma::mat Q, R;
             if(arma::qr(Q, R, X_out)) {
                 arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                 for (arma::uword k = 0; k < min_dim; ++k) if (R(k, k) < 0) Q.col(k) *= -1.0;
                 X_out = Q.cols(0, X_out.n_cols - 1);
             }
    } else {
        // Polar
        arma::mat U, V;
        arma::vec s;
        if (arma::svd_econ(U, s, V, X_out)) {
            X_out = U * V.t();
        } else {
             // Fallback to QR
             arma::mat Q, R;
             if(arma::qr(Q, R, X_out)) {
                 arma::uword min_dim = std::min(R.n_rows, R.n_cols);
                 for (arma::uword k = 0; k < min_dim; ++k) if (R(k, k) < 0) Q.col(k) *= -1.0;
                 X_out = Q.cols(0, X_out.n_cols - 1);
             }
        }
    }

    return X_out;
}

arma::mat OrbitalOptimizer::apply_preconditioner(const arma::mat& X, const arma::mat& grad) const {
    if (preconditioner_ == Preconditioner::None) {
        return grad;
    }

    arma::mat out = grad;
    const double eps = 1e-12;

    if (preconditioner_ == Preconditioner::ColumnNorm) {
        for (arma::uword j = 0; j < out.n_cols; ++j) {
            double nrm = arma::norm(out.col(j), 2);
            double scale = 1.0 / std::max(nrm, eps);
            out.col(j) *= scale;
        }
        return out;
    }

    // Diagonal-Hessian approximation: H_j ≈ mean(|g_j| / max(|x_j|, eps))
    for (arma::uword j = 0; j < out.n_cols; ++j) {
        arma::vec xcol = arma::abs(X.col(j));
        arma::vec gcol = arma::abs(out.col(j));
        xcol.transform([&](double v) { return std::max(v, eps); });
        arma::vec ratio = gcol / xcol;
        double hj = arma::mean(ratio);
        double scale = 1.0 / std::max(hj, eps);
        out.col(j) *= scale;
    }
    return out;
}

void OrbitalOptimizer::clear_history() {
    history_.clear();
}

void OrbitalOptimizer::update_history(const arma::mat& s, const arma::mat& y) {
    double sy = mat_dot(s, y);
    if (sy <= 1e-14) return;

    LbfgsPair pair;
    pair.s = s;
    pair.y = y;
    pair.rho = 1.0 / sy;

    history_.push_back(std::move(pair));
    if ((int)history_.size() > lbfgs_history_) {
        history_.erase(history_.begin());
    }
}

arma::mat OrbitalOptimizer::lbfgs_direction(const arma::mat& grad) const {
    if (history_.empty()) return -grad;

    arma::mat q = grad;
    std::vector<double> alpha(history_.size(), 0.0);

    for (int i = (int)history_.size() - 1; i >= 0; --i) {
        alpha[i] = history_[i].rho * mat_dot(history_[i].s, q);
        q -= alpha[i] * history_[i].y;
    }

    const LbfgsPair& last = history_.back();
    double yy = mat_dot(last.y, last.y);
    double sy = mat_dot(last.s, last.y);
    double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
    arma::mat r = gamma * q;

    for (size_t i = 0; i < history_.size(); ++i) {
        double beta = history_[i].rho * mat_dot(history_[i].y, r);
        r += history_[i].s * (alpha[i] - beta);
    }

    return -r;
}

OrbitalOptimizer::EvalResult OrbitalOptimizer::evaluate_point(
    const std::shared_ptr<EnergyFunctional<void>>& functional,
    const arma::mat& X_base,
    const arma::mat& dir,
    double alpha,
    const arma::vec& n,
    const arma::mat& S_inv_sqrt) const {
    EvalResult res;
    arma::mat X_trial = X_base + alpha * dir;
    res.X = retract(X_trial);
    res.C = from_orthogonal_basis(res.X, S_inv_sqrt);

    arma::mat gC;
    res.E = functional->energy(res.C, n);
    functional->orbital_gradient(res.C, n, gC);
    arma::mat gX = S_inv_sqrt * gC;
    res.grad = calc_riem_grad(res.X, gX);
    return res;
}

double OrbitalOptimizer::zoom_strong_wolfe(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                           const arma::mat& X,
                                           const arma::mat& dir,
                                           const arma::vec& n,
                                           const arma::mat& S_inv_sqrt,
                                           double E0,
                                           double dphi0,
                                           double a_lo,
                                           double a_hi,
                                           EvalResult& lo,
                                           EvalResult& hi,
                                           arma::mat& X_out,
                                           arma::mat& C_out,
                                           arma::mat& grad_out,
                                           double& E_out) const {
    const double c1 = 1e-4;
    const double c2 = 0.9;

    for (int iter = 0; iter < 10; ++iter) {
        double a = cubic_interpolate(a_lo, lo.E, mat_dot(lo.grad, dir),
                                     a_hi, hi.E, mat_dot(hi.grad, dir),
                                     std::min(a_lo, a_hi), std::max(a_lo, a_hi));

        EvalResult trial = evaluate_point(functional, X, dir, a, n, S_inv_sqrt);
        double dphi = mat_dot(trial.grad, dir);

        if ((trial.E > E0 + c1 * a * dphi0) || (trial.E >= lo.E)) {
            a_hi = a;
            hi = std::move(trial);
        } else {
            if (std::abs(dphi) <= -c2 * dphi0) {
                X_out = trial.X;
                C_out = trial.C;
                grad_out = trial.grad;
                E_out = trial.E;
                return a;
            }
            if (dphi * (a_hi - a_lo) >= 0.0) {
                a_hi = a_lo;
                hi = lo;
            }
            a_lo = a;
            lo = std::move(trial);
        }
    }

    X_out = lo.X;
    C_out = lo.C;
    grad_out = lo.grad;
    E_out = lo.E;
    return a_lo;
}

double OrbitalOptimizer::zoom_more_thuente(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                           const arma::mat& X,
                                           const arma::mat& dir,
                                           const arma::vec& n,
                                           const arma::mat& S_inv_sqrt,
                                           double E0,
                                           double dphi0,
                                           double a_lo,
                                           double a_hi,
                                           EvalResult& lo,
                                           EvalResult& hi,
                                           arma::mat& X_out,
                                           arma::mat& C_out,
                                           arma::mat& grad_out,
                                           double& E_out) const {
    const double c1 = 1e-4;
    const double c2 = 0.9;

    for (int iter = 0; iter < 25; ++iter) {
        double a = cubic_interpolate(a_lo, lo.E, mat_dot(lo.grad, dir),
                                     a_hi, hi.E, mat_dot(hi.grad, dir),
                                     std::min(a_lo, a_hi), std::max(a_lo, a_hi));

        EvalResult trial = evaluate_point(functional, X, dir, a, n, S_inv_sqrt);
        double dphi = mat_dot(trial.grad, dir);

        if ((trial.E > E0 + c1 * a * dphi0) || (trial.E >= lo.E)) {
            a_hi = a;
            hi = std::move(trial);
        } else {
            if (std::abs(dphi) <= -c2 * dphi0) {
                X_out = trial.X;
                C_out = trial.C;
                grad_out = trial.grad;
                E_out = trial.E;
                return a;
            }
            if (dphi * (a_hi - a_lo) >= 0.0) {
                a_hi = a_lo;
                hi = lo;
            }
            a_lo = a;
            lo = std::move(trial);
        }

        if (std::abs(a_hi - a_lo) < 1e-12) break;
    }

    X_out = lo.X;
    C_out = lo.C;
    grad_out = lo.grad;
    E_out = lo.E;
    return a_lo;
}

void OrbitalOptimizer::optimize(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                const arma::mat& S_sqrt,
                                const arma::mat& S_inv_sqrt,
                                arma::mat& C,
                                const arma::vec& n) {
    if (!functional) return;

    arma::mat X = to_orthogonal_basis(C, S_sqrt);

    arma::mat gC;
    double E = functional->energy(C, n);
    functional->orbital_gradient(C, n, gC);
    arma::mat gX = S_inv_sqrt * gC;
    arma::mat grad = calc_riem_grad(X, gX);
    
    arma::mat z = apply_preconditioner(X, grad);
    arma::mat dir = -z;
    clear_history();

    if (verbose_) {
        std::cout << "\n  --- Orbital Optimization ---\n";
        std::cout << "  Iter |      Energy      | Riem Grad Norm | Fidel. Err |  Step Size | Method\n";
        std::cout << "-------|------------------|----------------|------------|------------|--------\n";
    }

    arma::mat grad_prev = grad;
    arma::mat z_prev = z;
    arma::mat dir_prev = dir;

    for (int iter = 1; iter < max_iter_; ++iter) {
        double grad_norm = arma::norm(grad, "fro");
        
        double fid_err = 0.0;
        fid_err = arma::norm(X.t() * X - arma::eye(X.n_cols, X.n_cols), "inf");

        if (grad_norm < tol_) break;

        z = apply_preconditioner(X, grad);

        if (method_ == Method::SD) {
            dir = -z;
        } else if (method_ == Method::CG) {
            if (iter == 1) {
                dir = -z;
            } else {
                arma::mat z_trans = project_to_tangent(X, z_prev);
                double num = mat_dot(grad, z);
                double den = mat_dot(grad_prev, z_prev);
                double beta = std::max(0.0, (den > 0.0 ? num / den : 0.0));
                arma::mat dir_trans = project_to_tangent(X, dir_prev);
                dir = -z + beta * dir_trans;
                if (mat_dot(grad, dir) >= 0.0) dir = -z;
            }
        } else {
            if (iter == 1) {
                dir = -z;
            } else {
                dir = lbfgs_direction(grad);
                dir = project_to_tangent(X, dir);
                if (mat_dot(grad, dir) >= 0.0) dir = -z;
            }
        }

        double dphi0 = mat_dot(grad, dir);
        if (dphi0 >= 0.0) {
            dir = -grad;
            dphi0 = -mat_dot(grad, grad);
        }

        arma::mat X_new, C_new, grad_new;
        double E_new = E;
        double step = 0.0;

        // Adaptive step size guess
        double alpha_guess = 1.0;
        if (iter == 1) {
             alpha_guess = 0.2; // Conservative start
        } else if (step_size_ > 0.0) {
             alpha_guess = step_size_ * 2.0; 
        }
        
        if (alpha_guess < 1e-8) alpha_guess = 1e-4;
        if (alpha_guess > 10.0) alpha_guess = 1.0;

        if (line_search_ == LineSearch::StrongWolfe) {
            step = line_search_strong_wolfe(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        } else if (line_search_ == LineSearch::MoreThuente) {
            step = line_search_more_thuente(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        } else if (line_search_ == LineSearch::HagerZhang) {
            step = line_search_hager_zhang(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        } else if (line_search_ == LineSearch::Bracketing) {
            step = line_search_bracketing(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        } else if (line_search_ == LineSearch::Backtracking || line_search_ == LineSearch::Armijo) {
            step = line_search_armijo(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        } else {
            step = line_search_armijo(functional, X, dir, n, S_inv_sqrt, E, dphi0, alpha_guess, X_new, C_new, grad_new, E_new);
        }

        step_size_ = step; // Store for history


        if (verbose_) {
            std::cout << "  " << std::setw(4) << iter << " | "
                      << std::scientific << std::setprecision(10) << std::setw(16) << E
                      << " | " << std::scientific << std::setprecision(4) << std::setw(14) << grad_norm
                      << " | " << std::scientific << std::setprecision(4) << std::setw(10) << fid_err
                      << " | " << std::scientific << std::setprecision(4) << std::setw(10) << step
                      << " | ";

            if (method_ == Method::SD) std::cout << "SD\n";
            else if (method_ == Method::CG) std::cout << "CG\n";
            else std::cout << "L-BFGS\n";
        }

        if (step == 0.0) break;

        if (method_ == Method::LBFGS) {
            arma::mat step_vec = project_to_tangent(X_new, step * dir);
            arma::mat grad_trans = project_to_tangent(X_new, grad);
            arma::mat y = grad_new - grad_trans;
            update_history(step_vec, y);
        }

        grad_prev = grad;
        z_prev = z;
        dir_prev = dir;
        X = X_new;
        C = C_new;
        grad = grad_new;
        E = E_new;
    }

    if (verbose_) std::cout << "  [Orb] Energy after optimize: " << E << "\n";
}

double OrbitalOptimizer::line_search_armijo(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                            const arma::mat& X,
                                            const arma::mat& dir,
                                            const arma::vec& n,
                                            const arma::mat& S_inv_sqrt,
                                            double E0,
                                            double dphi0,
                                            double initial_alpha,
                                            arma::mat& X_out,
                                            arma::mat& C_out,
                                            arma::mat& grad_out,
                                            double& E_out) const {
    double alpha = initial_alpha;
    const double rho = 0.5;
    const double c1 = 1e-4;

    for (int i = 0; i < 20; ++i) {
        EvalResult trial = evaluate_point(functional, X, dir, alpha, n, S_inv_sqrt);
        if (trial.E <= E0 + c1 * alpha * dphi0) {
            X_out = std::move(trial.X);
            C_out = std::move(trial.C);
            grad_out = std::move(trial.grad);
            E_out = trial.E;
            return alpha;
        }
        alpha *= rho;
    }

    return 0.0;
}

double OrbitalOptimizer::line_search_bracketing(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                                const arma::mat& X,
                                                const arma::mat& dir,
                                                const arma::vec& n,
                                                const arma::mat& S_inv_sqrt,
                                                double E0,
                                                double dphi0,
                                                double initial_alpha,
                                                arma::mat& X_out,
                                                arma::mat& C_out,
                                                arma::mat& grad_out,
                                                double& E_out) const {
    const double c1 = 1e-4;
    const double max_alpha = 10.0;
    double alpha = initial_alpha;

    EvalResult prev = evaluate_point(functional, X, dir, 0.0, n, S_inv_sqrt);
    for (int i = 0; i < 20; ++i) {
        EvalResult cur = evaluate_point(functional, X, dir, alpha, n, S_inv_sqrt);
        double dphi = mat_dot(cur.grad, dir);

        if ((cur.E > E0 + c1 * alpha * dphi0) || (i > 0 && cur.E >= prev.E)) {
            return zoom_strong_wolfe(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, 0.0, alpha, prev, cur,
                                     X_out, C_out, grad_out, E_out);
        }

        if (std::abs(dphi) <= -0.9 * dphi0) {
            X_out = cur.X;
            C_out = cur.C;
            grad_out = cur.grad;
            E_out = cur.E;
            return alpha;
        }

        prev = std::move(cur);
        alpha = std::min(alpha * 2.0, max_alpha);
    }

    return 0.0;
}

double OrbitalOptimizer::line_search_strong_wolfe(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                                  const arma::mat& X,
                                                  const arma::mat& dir,
                                                  const arma::vec& n,
                                                  const arma::mat& S_inv_sqrt,
                                                  double E0,
                                                  double dphi0,
                                                  double initial_alpha,
                                                  arma::mat& X_out,
                                                  arma::mat& C_out,
                                                  arma::mat& grad_out,
                                                  double& E_out) const {
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const double max_alpha = 10.0;

    double alpha_prev = 0.0;
    EvalResult prev = evaluate_point(functional, X, dir, alpha_prev, n, S_inv_sqrt);

    double alpha = initial_alpha;

    for (int i = 0; i < 20; ++i) {
        EvalResult cur = evaluate_point(functional, X, dir, alpha, n, S_inv_sqrt);
        double dphi = mat_dot(cur.grad, dir);

        if ((cur.E > E0 + c1 * alpha * dphi0) || (i > 0 && cur.E >= prev.E)) {
            return zoom_strong_wolfe(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha_prev, alpha, prev, cur,
                                     X_out, C_out, grad_out, E_out);
        }

        if (std::abs(dphi) <= -c2 * dphi0) {
            X_out = cur.X;
            C_out = cur.C;
            grad_out = cur.grad;
            E_out = cur.E;
            return alpha;
        }

        if (dphi >= 0.0) {
            return zoom_strong_wolfe(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha, alpha_prev, cur, prev,
                                     X_out, C_out, grad_out, E_out);
        }

        alpha_prev = alpha;
        prev = std::move(cur);
        alpha = std::min(alpha * 2.0, max_alpha);
    }

    return 0.0;
}

double OrbitalOptimizer::line_search_more_thuente(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                                  const arma::mat& X,
                                                  const arma::mat& dir,
                                                  const arma::vec& n,
                                                  const arma::mat& S_inv_sqrt,
                                                  double E0,
                                                  double dphi0,
                                                  double initial_alpha,
                                                  arma::mat& X_out,
                                                  arma::mat& C_out,
                                                  arma::mat& grad_out,
                                                  double& E_out) const {
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const double max_alpha = 10.0;

    double alpha_prev = 0.0;
    EvalResult prev = evaluate_point(functional, X, dir, alpha_prev, n, S_inv_sqrt);

    double alpha = initial_alpha;

    for (int i = 0; i < 25; ++i) {
        EvalResult cur = evaluate_point(functional, X, dir, alpha, n, S_inv_sqrt);
        double dphi = mat_dot(cur.grad, dir);

        if ((cur.E > E0 + c1 * alpha * dphi0) || (i > 0 && cur.E >= prev.E)) {
            return zoom_more_thuente(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha_prev, alpha, prev, cur,
                                     X_out, C_out, grad_out, E_out);
        }

        if (std::abs(dphi) <= -c2 * dphi0) {
            X_out = cur.X;
            C_out = cur.C;
            grad_out = cur.grad;
            E_out = cur.E;
            return alpha;
        }

        if (dphi >= 0.0) {
            return zoom_more_thuente(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha, alpha_prev, cur, prev,
                                     X_out, C_out, grad_out, E_out);
        }

        alpha_prev = alpha;
        prev = std::move(cur);
        alpha = std::min(alpha * 1.6, max_alpha);
    }

    return 0.0;
}

double OrbitalOptimizer::line_search_hager_zhang(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                                 const arma::mat& X,
                                                 const arma::mat& dir,
                                                 const arma::vec& n,
                                                 const arma::mat& S_inv_sqrt,
                                                 double E0,
                                                 double dphi0,
                                                 double initial_alpha,
                                                 arma::mat& X_out,
                                                 arma::mat& C_out,
                                                 arma::mat& grad_out,
                                                 double& E_out) const {
    // Practical Hager-Zhang-style strong Wolfe with safeguarded bracketing
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const double max_alpha = 10.0;

    double alpha_prev = 0.0;
    EvalResult prev = evaluate_point(functional, X, dir, alpha_prev, n, S_inv_sqrt);
    double alpha = initial_alpha;

    for (int i = 0; i < 25; ++i) {
        EvalResult cur = evaluate_point(functional, X, dir, alpha, n, S_inv_sqrt);
        double dphi = mat_dot(cur.grad, dir);

        if ((cur.E > E0 + c1 * alpha * dphi0) || (i > 0 && cur.E >= prev.E)) {
            return zoom_strong_wolfe(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha_prev, alpha, prev, cur,
                                     X_out, C_out, grad_out, E_out);
        }

        if (std::abs(dphi) <= -c2 * dphi0) {
            X_out = cur.X;
            C_out = cur.C;
            grad_out = cur.grad;
            E_out = cur.E;
            return alpha;
        }

        if (dphi >= 0.0) {
            return zoom_strong_wolfe(functional, X, dir, n, S_inv_sqrt,
                                     E0, dphi0, alpha, alpha_prev, cur, prev,
                                     X_out, C_out, grad_out, E_out);
        }

        alpha_prev = alpha;
        prev = std::move(cur);
        alpha = std::min(alpha * 1.6, max_alpha);
    }

    return 0.0;
}

} // namespace rdmft
} // namespace helfem
