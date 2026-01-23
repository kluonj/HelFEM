#ifndef HELFEM_RDMFT_OPTIMIZER_H
#define HELFEM_RDMFT_OPTIMIZER_H

#include <armadillo>
#include <memory>
#include <vector>
#include <algorithm>
#include "energy.h"

namespace helfem {
namespace rdmft {

class OrbitalOptimizer {
public:
    enum class Method {
        SD,
        CG,
        LBFGS
    };

    enum class LineSearch {
        Backtracking,
        Bracketing,
        Armijo,
        StrongWolfe,
        MoreThuente,
        HagerZhang
    };

    enum class Preconditioner {
        None,
        ColumnNorm,
        DiagHessian
    };

    OrbitalOptimizer() = default;

    void set_method(Method m) { method_ = m; }
    void set_line_search(LineSearch m) { line_search_ = m; }
    void set_lbfgs_history(int m) { lbfgs_history_ = std::max(1, m); }
    void set_max_iter(int n) { max_iter_ = n; }
    void set_tol(double t) { tol_ = t; }
    void set_verbose(bool v) { verbose_ = v; }
    void set_preconditioner(Preconditioner p) { preconditioner_ = p; }

    void optimize(const std::shared_ptr<EnergyFunctional<void>>& functional,
                  const arma::mat& S_sqrt,
                  const arma::mat& S_inv_sqrt,
                  arma::mat& C,
                  const arma::vec& n,
                  int n_alpha_orb);

private:
    struct EvalResult;
    struct LbfgsPair {
        arma::mat s;
        arma::mat y;
        double rho = 0.0;
    };

    Method method_ = Method::CG;
    LineSearch line_search_ = LineSearch::Armijo;
    Preconditioner preconditioner_ = Preconditioner::None;
    int lbfgs_history_ = 6;
    int max_iter_ = 20;
    double tol_ = 1e-6;
    bool verbose_ = true;

    std::vector<LbfgsPair> history_;

    arma::mat to_orthogonal_basis(const arma::mat& C, const arma::mat& S_sqrt) const;
    arma::mat from_orthogonal_basis(const arma::mat& X, const arma::mat& S_inv_sqrt) const;

    arma::mat calc_riem_grad(const arma::mat& X, const arma::mat& gX, int n_alpha_orb) const;
    arma::mat project_to_tangent(const arma::mat& X, const arma::mat& V, int n_alpha_orb) const;
    arma::mat retract(const arma::mat& X_trial, int n_alpha_orb) const;
    arma::mat apply_preconditioner(const arma::mat& X, const arma::mat& grad) const;

    EvalResult evaluate_point(const std::shared_ptr<EnergyFunctional<void>>& functional,
                              const arma::mat& X_base,
                              const arma::mat& dir,
                              double alpha,
                              const arma::vec& n,
                              const arma::mat& S_inv_sqrt,
                              int n_alpha_orb) const;

    double zoom_strong_wolfe(const std::shared_ptr<EnergyFunctional<void>>& functional,
                             const arma::mat& X,
                             const arma::mat& dir,
                             const arma::vec& n,
                             const arma::mat& S_inv_sqrt,
                             int n_alpha_orb,
                             double E0,
                             double dphi0,
                             double a_lo,
                             double a_hi,
                             EvalResult& lo,
                             EvalResult& hi,
                             arma::mat& X_out,
                             arma::mat& C_out,
                             arma::mat& grad_out,
                             double& E_out) const;

    double zoom_more_thuente(const std::shared_ptr<EnergyFunctional<void>>& functional,
                             const arma::mat& X,
                             const arma::mat& dir,
                             const arma::vec& n,
                             const arma::mat& S_inv_sqrt,
                             int n_alpha_orb,
                             double E0,
                             double dphi0,
                             double a_lo,
                             double a_hi,
                             EvalResult& lo,
                             EvalResult& hi,
                             arma::mat& X_out,
                             arma::mat& C_out,
                             arma::mat& grad_out,
                             double& E_out) const;

    double line_search_armijo(const std::shared_ptr<EnergyFunctional<void>>& functional,
                              const arma::mat& X,
                              const arma::mat& dir,
                              const arma::vec& n,
                              const arma::mat& S_inv_sqrt,
                              int n_alpha_orb,
                              double E0,
                              double dphi0,
                              arma::mat& X_out,
                              arma::mat& C_out,
                              arma::mat& grad_out,
                              double& E_out) const;

    double line_search_bracketing(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                  const arma::mat& X,
                                  const arma::mat& dir,
                                  const arma::vec& n,
                                  const arma::mat& S_inv_sqrt,
                                  int n_alpha_orb,
                                  double E0,
                                  double dphi0,
                                  arma::mat& X_out,
                                  arma::mat& C_out,
                                  arma::mat& grad_out,
                                  double& E_out) const;

    double line_search_strong_wolfe(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                    const arma::mat& X,
                                    const arma::mat& dir,
                                    const arma::vec& n,
                                    const arma::mat& S_inv_sqrt,
                                    int n_alpha_orb,
                                    double E0,
                                    double dphi0,
                                    arma::mat& X_out,
                                    arma::mat& C_out,
                                    arma::mat& grad_out,
                                    double& E_out) const;

    double line_search_more_thuente(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                    const arma::mat& X,
                                    const arma::mat& dir,
                                    const arma::vec& n,
                                    const arma::mat& S_inv_sqrt,
                                    int n_alpha_orb,
                                    double E0,
                                    double dphi0,
                                    arma::mat& X_out,
                                    arma::mat& C_out,
                                    arma::mat& grad_out,
                                    double& E_out) const;

    double line_search_hager_zhang(const std::shared_ptr<EnergyFunctional<void>>& functional,
                                   const arma::mat& X,
                                   const arma::mat& dir,
                                   const arma::vec& n,
                                   const arma::mat& S_inv_sqrt,
                                   int n_alpha_orb,
                                   double E0,
                                   double dphi0,
                                   arma::mat& X_out,
                                   arma::mat& C_out,
                                   arma::mat& grad_out,
                                   double& E_out) const;

    void clear_history();
    void update_history(const arma::mat& s, const arma::mat& y);
    arma::mat lbfgs_direction(const arma::mat& grad) const;
};

} // namespace rdmft
} // namespace helfem

#endif
