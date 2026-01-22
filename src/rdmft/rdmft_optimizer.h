#ifndef HELFEM_RDMFT_OPTIMIZER_H
#define HELFEM_RDMFT_OPTIMIZER_H

#include <armadillo>
#include <functional>
#include <memory>
#include <string>

namespace helfem {
namespace rdmft {

// Abstract base class for optimization algorithms
class Optimizer {
public:
    virtual ~Optimizer() = default;

    struct State {
        arma::mat C; // Current orbitals
        arma::vec n; // Current occupations
        double energy;
        arma::mat gC; // Gradient w.r.t orbitals (Riemannian if apply_manifold is used)
        arma::vec gn; // Gradient w.r.t occupations
    };

    // Callback function type for energy and gradient evaluation
    // Returns energy, updates gradients in state (Euclidean gradients)
    using EnergyGradFunc = std::function<double(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn)>;
    
    // Callback for applying manifold constraints / Riemannian geometry
    // 1. Project Euclidean gradient to Riemannian gradient
    using ManifoldGradFunc = std::function<void(const arma::mat& C, const arma::vec& n, arma::mat& gC, arma::vec& gn)>;
    
    // 2. Retraction: Move point C along direction dir by amount alpha
    using RetractFunc = std::function<void(const State& current, const arma::mat& dir_C, const arma::vec& dir_n, double alpha, State& next)>;

    // 3. Vector Transport: Transport vector 'vec' from old_state to new_state
    using TransportFunc = std::function<void(const State& old_state, const State& new_state, arma::mat& vec_C, arma::vec& vec_n)>;
    
    // 4. Inner Product: Compute <v1, v2>_x in the metric of the manifold
    using InnerProductFunc = std::function<double(const State& state, const arma::mat& v1_C, const arma::vec& v1_n, const arma::mat& v2_C, const arma::vec& v2_n)>;

    enum class InitialStep {
        Fixed,       // Always 1.0 (or fixed value)
        InverseNorm, // 1.0 / |g|
        BarzilaiBorwein, // BB1 or BB2
        Quadratic    // Quadratic interpolation
    };

    // Configuration
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_step_tol(double tol) { step_tol_ = tol; }
    void set_grad_tol(double tol) { grad_tol_ = tol; }
    void set_energy_tol(double tol) { energy_tol_ = tol; }
    void set_verbose(bool v) { verbose_ = v; }
    void set_initial_step(InitialStep type) { init_step_type_ = type; }

    // Optimization interface
    virtual void optimize(State& state, 
                          EnergyGradFunc energy_func, 
                          ManifoldGradFunc manifold_grad_func = nullptr,
                          RetractFunc retract_func = nullptr,
                          TransportFunc transport_func = nullptr,
                          InnerProductFunc inner_product_func = nullptr) = 0;

protected:
    int max_iter_ = 20;
    double step_tol_ = 1e-6;
    double grad_tol_ = 1e-5;
    double energy_tol_ = 1e-7;
    bool verbose_ = false;
    InitialStep init_step_type_ = InitialStep::Quadratic; // Default to old behavior equivalent
};

// Steepest Descent Optimizer
class SteepestDescentOptimizer : public Optimizer {
public:
    void optimize(State& state, 
                  EnergyGradFunc energy_func, 
                  ManifoldGradFunc manifold_grad_func = nullptr,
                  RetractFunc retract_func = nullptr,
                  TransportFunc transport_func = nullptr,
                  InnerProductFunc inner_product_func = nullptr) override;
};

// Conjugate Gradient Optimizer (Polak-Ribiere)
class ConjugateGradientOptimizer : public Optimizer {
public:
    void optimize(State& state, 
                  EnergyGradFunc energy_func, 
                  ManifoldGradFunc manifold_grad_func = nullptr,
                  RetractFunc retract_func = nullptr,
                  TransportFunc transport_func = nullptr,
                  InnerProductFunc inner_product_func = nullptr) override;
};

// BFGS Optimizer (Simplified / Limited Memory placeholder)
class BFGSOptimizer : public Optimizer {
public:
    void optimize(State& state, 
                  EnergyGradFunc energy_func, 
                  ManifoldGradFunc manifold_grad_func = nullptr,
                  RetractFunc retract_func = nullptr,
                  TransportFunc transport_func = nullptr,
                  InnerProductFunc inner_product_func = nullptr) override;
};


} // namespace rdmft
} // namespace helfem

#endif // HELFEM_RDMFT_OPTIMIZER_H
