#include "rdmft_optimizer.h"

#include "../atomic/basis.h"
#include "../general/model_potential.h"
#include "../../libhelfem/include/PolynomialBasis.h"
#include "../general/scf_helpers.h"

#include <armadillo>
#include <algorithm>
#include <iostream>

// Compact RDMFT test for a Helium atom using the Muller functional.
// Uses a small basis so the test completes quickly on CI / local machines.

int main(int /*argc*/, char** /*argv*/) {
  using namespace helfem::atomic::basis;

  // Helium atom
  int Z = 2;
  helfem::modelpotential::nuclear_model_t model = helfem::modelpotential::POINT_NUCLEUS;

  // Small basis parameters to keep the test fast
  double Rrms = 1.0;
  int primbas = 4;   // use 4 primitives
  int Nnodes = 8;    // small number of LIP nodes
  bool zeroder = false;
  int Nquad = 0;
  int lmax = 0;
  int mmax = 0;

  auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
  if (Nquad == 0) Nquad = 5 * std::max(1, poly->get_nbf());

  arma::ivec lval, mval;
  helfem::atomic::basis::angular_basis(lmax, mmax, lval, mval);

  int Nelem = 3;
  double Rmax = 20.0;
  int igrid = 3;
  double zexp = 2.0;
  int Nelem0 = 0;
  int igrid0 = 3;
  double zexp0 = 2.0;
  arma::vec bval = helfem::atomic::basis::form_grid(model, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, 0, 0, 0.0);

  int taylor_order = std::max(0, poly->get_nprim() - 1);

  TwoDBasis basis;
  try {
    basis = TwoDBasis(Z, model, Rrms, poly, zeroder, Nquad, bval, taylor_order, lval, mval, /*Zl=*/0, /*Zr=*/0, /*Rhalf=*/0.0);
  } catch (std::exception &e) {
    std::cerr << "Failed to construct basis: " << e.what() << "\n";
    return 1;
  }

  double Nelec = 2.0;
  size_t Norb_desired = 2; // restrict to 2 natural orbitals for He
  double power = 0.5; // Muller

  arma::mat C;
  arma::vec n;

  helfem::rdmft::Options opt;
  opt.algorithm = helfem::rdmft::Algorithm::ConjugateGradient;
  opt.max_iter = 300;
  opt.grad_tol = 1e-6;
  opt.step0 = 0.1;

  try {
    std::cout << "basis nbf=" << basis.Nbf() << "\n";
    size_t Norb = std::min(Norb_desired, static_cast<size_t>(basis.Nbf()));
    std::cout << "using Norb=" << Norb << " (desired " << Norb_desired << ")\n";
    std::cout << "overlap size=" << basis.overlap().n_rows << "x" << basis.overlap().n_cols << "\n";

    // Ensure TEIs are available for hartree/exchange
    ::detail::ensure_tei(basis);

    // Build initial guess (generalized eigenvectors of Hcore)
    arma::mat S = basis.overlap();
    arma::mat Hcore = basis.kinetic() + basis.nuclear();
    arma::mat Sinvh = helfem::scf::form_Sinvh(S, /*chol=*/false);
    arma::vec eps; arma::mat evec;
    arma::mat Horth = arma::trans(Sinvh) * Hcore * Sinvh;
    arma::eig_sym(eps, evec, Horth);
    evec = Sinvh * evec;
    if (evec.n_cols < Norb) throw std::logic_error("Basis too small for requested Norb");
    C = evec.cols(0, Norb - 1);

    n.set_size(Norb);
    n.fill(std::min(1.0, Nelec / double(Norb)));
    n = helfem::rdmft::project_capped_simplex(n, Nelec);

    arma::vec theta = arma::acos(arma::sqrt(n));

    // Local Muller adapter using the component functions
    struct MullerAdapter : public helfem::rdmft::EnergyFunctional<TwoDBasis> {
      MullerAdapter(TwoDBasis& b, const arma::mat& H0, double p) : basis_(b), Hcore_(H0), power_(p) {}
      double energy(const arma::mat& C_AO, const arma::vec& n, arma::mat& gC_AO, arma::vec& gn) override {
        double E_core = helfem::rdmft::core_energy<TwoDBasis>(Hcore_, C_AO, n);
        double E_J = helfem::rdmft::hartree_energy<TwoDBasis>(basis_, C_AO, n);
        double E_xc = helfem::rdmft::muller_xc_energy<TwoDBasis>(basis_, C_AO, n, power_);
        arma::mat gC_core; helfem::rdmft::core_orbital_gradient<TwoDBasis>(Hcore_, C_AO, n, gC_core);
        arma::mat gC_J; helfem::rdmft::hartree_orbital_gradient<TwoDBasis>(basis_, C_AO, n, gC_J);
        arma::mat gC_xc; helfem::rdmft::muller_xc_orbital_gradient<TwoDBasis>(basis_, C_AO, n, power_, gC_xc);
        gC_AO = gC_core + gC_J + gC_xc;
        arma::vec gn_core; helfem::rdmft::core_occupation_gradient<TwoDBasis>(Hcore_, C_AO, n, gn_core);
        arma::vec gn_J; helfem::rdmft::hartree_occupation_gradient<TwoDBasis>(basis_, C_AO, n, gn_J);
        arma::vec gn_xc; helfem::rdmft::muller_occupation_gradient<TwoDBasis>(basis_, C_AO, n, power_, gn_xc);
        arma::uword Ne = std::max(gn_core.n_elem, std::max(gn_J.n_elem, gn_xc.n_elem));
        gn.reset(); gn.set_size(Ne); gn.zeros();
        if (gn_core.n_elem) gn.head(gn_core.n_elem) += gn_core;
        if (gn_J.n_elem) gn.head(gn_J.n_elem) += gn_J;
        if (gn_xc.n_elem) gn.head(gn_xc.n_elem) += gn_xc;
        return E_core + E_J + E_xc;
      }
      TwoDBasis& basis_;
      arma::mat Hcore_;
      double power_;
    } adapter(basis, Hcore, power);

    // Run joint optimizer directly
    helfem::rdmft::JointOptimizer<helfem::rdmft::EnergyFunctional<TwoDBasis>> optimizer(S, adapter, Nelec, opt);
    auto res = optimizer.minimize(C, theta);
    n = helfem::rdmft::occ_from_theta(theta);
    if (opt.enforce_sum_projection) n = helfem::rdmft::project_capped_simplex(n, Nelec);

    std::cout << "RDMFT Muller test (He)\n";
    std::cout << "iters=" << res.iters << " converged=" << res.converged << "\n";
    std::cout << "E=" << res.energy << "\n";
    std::cout << "|grad|=" << res.grad_norm << "\n";
    std::cout << "sum(n)=" << arma::sum(n) << " (target " << Nelec << ")\n";
    std::cout << "min(n)=" << n.min() << " max(n)=" << n.max() << "\n";
    std::cout << "orth_err=" << helfem::manifold::orthonormality_error(C, basis.overlap()) << "\n";

    if (!res.converged) return 2;
    if (std::abs(arma::sum(n) - Nelec) > 1e-6) return 3;
    if (helfem::manifold::orthonormality_error(C, basis.overlap()) > 1e-6) return 4;
    if (n.min() < -1e-12 || n.max() > 1.0 + 1e-12) return 5;
  } catch (std::exception &e) {
    std::cerr << "RDMFT Muller test failed: " << e.what() << "\n";
    return 6;
  }

  return 0;
}
