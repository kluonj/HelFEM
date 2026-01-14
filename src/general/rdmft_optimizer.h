#ifndef HELFEM_RDMFT_OPTIMIZER_H
#define HELFEM_RDMFT_OPTIMIZER_H

#include "generalized_stiefel.h"
#include "lbfgs.h"

#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace helfem {
namespace rdmft {

enum class Algorithm {
  SteepestDescent,
  ConjugateGradient,
  BFGS
};

struct Options {
  Algorithm algorithm = Algorithm::ConjugateGradient;
  size_t max_iter = 200;
  double grad_tol = 1e-6;
  double step0 = 1.0;
  double ls_c1 = 1e-4;
  double ls_shrink = 0.5;
  size_t ls_max_steps = 25;

  // Occupations: if true, enforce 0<=n<=1 and sum(n)=N exactly
  // by projecting n=cos^2(theta) onto the capped simplex after each step.
  // This is a projected-gradient style approach (n-constraints are not a smooth manifold at the box boundaries).
  bool enforce_sum_projection = true;

  // Augmented Lagrangian for sum(n)=Ne.
  double aug_mu0 = 10.0;
  double aug_mu_max = 1e6;
  double aug_mu_growth = 5.0;
  double aug_constraint_tol = 1e-10;
};

struct Result {
  bool converged = false;
  size_t iters = 0;
  double energy = std::numeric_limits<double>::quiet_NaN();
  double grad_norm = std::numeric_limits<double>::quiet_NaN();
  double constraint = std::numeric_limits<double>::quiet_NaN();
};

inline arma::vec occ_from_theta(const arma::vec& theta) {
  // Elementwise cos^2(theta) in [0,1]
  return arma::square(arma::cos(theta));
}

inline arma::vec d_occ_d_theta(const arma::vec& theta) {
  // d/dtheta cos^2(theta) = -sin(2*theta)
  return -arma::sin(2.0 * theta);
}

inline double product_inner(const arma::mat& A, const arma::mat& B, const arma::mat& S, const arma::vec& a, const arma::vec& b) {
  return helfem::manifold::inner_product(A, B, S) + arma::dot(a, b);
}

inline double product_norm(const arma::mat& A, const arma::mat& S, const arma::vec& a) {
  return std::sqrt(product_inner(A, A, S, a, a));
}

inline arma::vec project_capped_simplex(const arma::vec& y, double target_sum) {
  // Projection onto {x : 0 <= x_i <= 1, sum x_i = target_sum}
  // via bisection on Lagrange multiplier: x_i = clamp(y_i - lambda, 0, 1)
  if (!y.is_finite()) throw std::logic_error("project_capped_simplex: non-finite input");
  if (target_sum < 0.0) throw std::logic_error("project_capped_simplex: target_sum < 0");
  if (target_sum > double(y.n_elem)) throw std::logic_error("project_capped_simplex: target_sum too large");

  auto sum_clamped = [&](double lambda) {
    arma::vec x = y - lambda;
    x.transform([](double v) { return std::min(1.0, std::max(0.0, v)); });
    return arma::sum(x);
  };

  // Bracket lambda.
  double lo = y.min() - 1.0;
  double hi = y.max();
  // Ensure monotone bracket: sum(lo) >= target, sum(hi) <= target
  if (sum_clamped(lo) < target_sum) {
    lo = lo - 10.0;
  }
  if (sum_clamped(hi) > target_sum) {
    hi = hi + 10.0;
  }

  for (int it = 0; it < 80; ++it) {
    double mid = 0.5 * (lo + hi);
    double s = sum_clamped(mid);
    if (s > target_sum) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  arma::vec x = y - 0.5 * (lo + hi);
  x.transform([](double v) { return std::min(1.0, std::max(0.0, v)); });

  // Small numerical correction to hit target_sum in case of rounding.
  double diff = arma::sum(x) - target_sum;
  if (std::abs(diff) > 1e-10) {
    // distribute diff over non-saturated entries
    arma::uvec free = arma::find((x > 1e-12) % (x < 1.0 - 1e-12));
    if (!free.empty()) {
      x(free) -= diff / double(free.n_elem);
      x.transform([](double v) { return std::min(1.0, std::max(0.0, v)); });
    }
  }
  return x;
}

// Functional concept:
//   double func(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn)
// where gC and gn are Euclidean gradients.

template <typename Functional>
class JointOptimizer {
public:
  JointOptimizer(const arma::mat& S, Functional& functional, double Nelec, const Options& opt)
      : S_(S), functional_(functional), Nelec_(Nelec), opt_(opt) {
    if (S_.n_rows != S_.n_cols) throw std::logic_error("S must be square");
    if (Nelec_ <= 0.0) throw std::logic_error("Nelec must be positive");
  }

  Result minimize(arma::mat& C, arma::vec& theta) {
    if (C.n_rows != S_.n_rows) throw std::logic_error("C and S dimension mismatch");
    if (theta.n_elem == 0) throw std::logic_error("theta is empty");

    // Enforce starting point on manifold.
    C = helfem::manifold::retract_chol(C, arma::zeros(C.n_rows, C.n_cols), S_);

    double lambda = 0.0;
    double mu = opt_.aug_mu0;
    double c_abs_prev = std::numeric_limits<double>::infinity();
    size_t c_bad_steps = 0;

    arma::mat gC_eu, gC;
    arma::vec gn, gtheta;

    arma::mat dirC(C.n_rows, C.n_cols, arma::fill::zeros);
    arma::vec dirTheta(theta.n_elem, arma::fill::zeros);

    arma::mat gC_prev;
    arma::vec gtheta_prev;

    LBFGS bfgs;
    arma::vec x_prev;

    Result res;

    for (size_t iter = 0; iter < opt_.max_iter; ++iter) {
      arma::vec n = occ_from_theta(theta);
      if (opt_.enforce_sum_projection) {
        n = project_capped_simplex(n, Nelec_);
        // Keep theta consistent with projected occupations
        theta = arma::acos(arma::sqrt(n));
      }
      double c = arma::sum(n) - Nelec_;

      // Energy and gradients
      double E = functional_(C, n, gC_eu, gn);

      if (!opt_.enforce_sum_projection) {
        // Augmented Lagrangian contribution
        E += lambda * c + 0.5 * mu * c * c;
        arma::vec gn_aug = gn + (lambda + mu * c) * arma::ones(gn.n_elem);
        gtheta = gn_aug % d_occ_d_theta(theta);
      } else {
        // Projected occupations: use projected-gradient style update.
        gtheta = gn % d_occ_d_theta(theta);
      }

      // Riemannian gradient on generalized Stiefel
      gC = helfem::manifold::project_tangent(C, gC_eu, S_);

      double gnorm = product_norm(gC, S_, gtheta);

      res.energy = E;
      res.grad_norm = gnorm;
      res.constraint = c;
      res.iters = iter + 1;

      // Convergence: gradient and constraint
      double c_tol = opt_.enforce_sum_projection ? 1e-8 : opt_.aug_constraint_tol;
      if (gnorm < opt_.grad_tol && std::abs(c) < c_tol) {
        res.converged = true;
        break;
      }

      // Choose search direction
      if (opt_.algorithm == Algorithm::SteepestDescent) {
        dirC = -gC;
        dirTheta = -gtheta;
      } else if (opt_.algorithm == Algorithm::ConjugateGradient) {
        if (iter == 0) {
          dirC = -gC;
          dirTheta = -gtheta;
        } else {
          // Transport previous direction to current tangent by projection
          arma::mat dirC_tr = helfem::manifold::project_tangent(C, dirC, S_);

          arma::mat yC = gC - gC_prev;
          arma::vec yT = gtheta - gtheta_prev;

          double denom = product_inner(gC_prev, gC_prev, S_, gtheta_prev, gtheta_prev);
          double beta = 0.0;
          if (denom > 1e-18) {
            beta = product_inner(gC, yC, S_, gtheta, yT) / denom; // Polak-Ribiere
          }
          beta = std::max(0.0, beta);

          dirC = -gC + beta * dirC_tr;
          dirTheta = -gtheta + beta * dirTheta;
        }
      } else { // BFGS
        arma::vec x = arma::join_vert(arma::vectorise(C), theta);
        arma::vec g = arma::join_vert(arma::vectorise(gC), gtheta);
        if (iter == 0) {
          bfgs.clear();
        }
        bfgs.update(x, g);
        arma::vec p = -bfgs.solve();
        if (!p.is_finite()) {
          // Fallback
          dirC = -gC;
          dirTheta = -gtheta;
        } else {
          arma::vec pC = p.head(C.n_elem);
          arma::vec pT = p.tail(theta.n_elem);
          dirC = arma::reshape(pC, C.n_rows, C.n_cols);
          dirC = helfem::manifold::project_tangent(C, dirC, S_);
          dirTheta = pT;
        }
        x_prev = x;
      }

      // Armijo backtracking line search on augmented objective
      double t = opt_.step0;
      double gdotp = product_inner(gC, dirC, S_, gtheta, dirTheta);
      // Ensure descent; if not, fallback to steepest descent
      if (gdotp > 0.0) {
        dirC = -gC;
        dirTheta = -gtheta;
        gdotp = -product_inner(gC, gC, S_, gtheta, gtheta);
      }

      arma::mat C_best = C;
      arma::vec theta_best = theta;
      double E_best = E;

      bool accepted = false;
      for (size_t ls = 0; ls < opt_.ls_max_steps; ++ls) {
        arma::mat C_trial = helfem::manifold::retract_chol(C, t * dirC, S_);
        arma::vec theta_trial = theta + t * dirTheta;

        arma::vec n_trial = occ_from_theta(theta_trial);
        if (opt_.enforce_sum_projection) {
          n_trial = project_capped_simplex(n_trial, Nelec_);
          theta_trial = arma::acos(arma::sqrt(n_trial));
        }
        double c_trial = arma::sum(n_trial) - Nelec_;

        arma::mat gC_tmp;
        arma::vec gn_tmp;
        double E_trial = functional_(C_trial, n_trial, gC_tmp, gn_tmp);
        if (!opt_.enforce_sum_projection) {
          E_trial += lambda * c_trial + 0.5 * mu * c_trial * c_trial;
        }

        if (E_trial <= E + opt_.ls_c1 * t * gdotp) {
          C_best = std::move(C_trial);
          theta_best = std::move(theta_trial);
          E_best = E_trial;
          accepted = true;
          break;
        }
        t *= opt_.ls_shrink;
      }

      if (!accepted) {
        // If line search failed, take a tiny step to avoid stalling.
        t = 1e-3;
        C_best = helfem::manifold::retract_chol(C, t * (-gC), S_);
        theta_best = theta + t * (-gtheta);
      }

      C = std::move(C_best);
      theta = std::move(theta_best);
      res.energy = E_best;

      if (!opt_.enforce_sum_projection) {
        // Update multipliers (Augmented Lagrangian)
        arma::vec n_new = occ_from_theta(theta);
        double c_new = arma::sum(n_new) - Nelec_;
        lambda += mu * c_new;

        // Increase penalty only if constraint stagnates for several iterations.
        double c_abs = std::abs(c_new);
        if (c_abs_prev < std::numeric_limits<double>::infinity()) {
          if (c_abs > 0.9 * c_abs_prev) {
            ++c_bad_steps;
          } else {
            c_bad_steps = 0;
          }
        }
        c_abs_prev = c_abs;

        if (c_bad_steps >= 5 && mu < opt_.aug_mu_max) {
          mu = std::min(opt_.aug_mu_max, mu * opt_.aug_mu_growth);
          c_bad_steps = 0;
        }
      }

      gC_prev = gC;
      gtheta_prev = gtheta;
    }

    return res;
  }

private:
  arma::mat S_;
  Functional& functional_;
  double Nelec_;
  Options opt_;
};

} // namespace rdmft
} // namespace helfem

#endif
