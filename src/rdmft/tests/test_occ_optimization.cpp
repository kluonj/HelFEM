// test_occ_optimization.cpp
// Simple unit test for occupation optimization step in Solver.

#include <iostream>
#include <armadillo>
#include "test_utils.h"
#include "rdmft/solver.h"
#include "rdmft/energy.h"
#include "rdmft/gradients.h"

using namespace helfem;
using namespace helfem::rdmft;

// Quadratic functional that depends only on occupations n:
// E(n) = sum_i a_i * (n_i - t_i)^2
struct QuadraticOccFunctional : public EnergyFunctional<void> {
    arma::vec t;
    arma::vec a;
    QuadraticOccFunctional(const arma::vec& target, const arma::vec& alpha) : t(target), a(alpha) {}

    double energy(const arma::mat& C, const arma::vec& n) override {
        arma::vec diff = n - t;
        return arma::dot(a % diff % diff, arma::ones<arma::vec>(diff.n_elem));
    }

    void orbital_gradient(const arma::mat& C, const arma::vec& n, arma::mat& gC) override {
        gC = arma::mat(C.n_rows, C.n_cols, arma::fill::zeros);
    }

    void occupation_gradient(const arma::mat& C, const arma::vec& n, arma::vec& gn) override {
        gn = 2.0 * (a % (n - t));
    }
};

int main() {
    int m = 6;
    arma::mat S = arma::eye(m, m);
    arma::mat C = arma::eye<arma::mat>(m, m);

    // Choose a target inside (0,1) so the constrained minimizer is interior
    arma::vec t = {0.4, 0.6, 0.3, 0.2, 0.5, 0.5};
    arma::vec a = arma::ones(m) * 1.0;
    double N = arma::sum(t);

    auto func = std::make_shared<QuadraticOccFunctional>(t, a);
    Solver solver(func, S);
    solver.set_verbose(true);
    solver.set_optimize_orbitals(false);
    solver.set_optimize_occupations(true);
    solver.set_max_occ_iter(200);
    solver.set_occ_tol(1e-12);

    arma::vec n = arma::vec(m).fill(N / double(m));

    arma::vec expected = project_capped_simplex(t, N);

    solver.solve(C, n, N);

    double err = arma::norm(n - expected, "inf");
    std::cout << "final n: " << n.t();
    std::cout << "expected: " << expected.t();
    std::cout << "inf-norm error: " << err << std::endl;
    if (err > 1e-8) {
        std::cerr << "Occupation optimization test FAILED\n";
        return 2;
    }
    std::cout << "Occupation optimization test PASSED\n";
    return 0;
}
