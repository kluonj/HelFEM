#include "../general/checkpoint.h"
#include "../rdmft/rdmft_energy.h"
#include "../rdmft/rdmft_gradients.h"
#include "../atomic/TwoDBasis.h"

#include <armadillo>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string find_atomic() {
  std::vector<std::string> candidates = {
    "./atomic",
    "./src/atomic",
    "build/src/atomic",
    "../build/src/atomic",
    "/home/kluo/Documents/repo/HelFEM/build/src/atomic"
  };
  for (const auto &c : candidates) {
    std::ifstream f(c);
    if (f.good()) return c;
  }
  return "";
}

bool file_exists_local(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

double total_energy(helfem::atomic::basis::TwoDBasis &basis,
                    const arma::mat &C_AO,
                    const arma::vec &nocc,
                    double power) {
  arma::mat Hcore = basis.kinetic() + basis.nuclear();
  double Ecore = helfem::rdmft::core_energy<helfem::atomic::basis::TwoDBasis>(Hcore, C_AO, nocc);
  double EJ = helfem::rdmft::hartree_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc);
  double Exc = helfem::rdmft::muller_xc_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, power);
  return Ecore + EJ + Exc;
}
}

int main() {
  try {
    const std::string outfn = "atomic.out";
    std::string atomic_path = find_atomic();
    if (!atomic_path.empty() && std::getenv("HAVE_ATOMIC") != nullptr) {
      std::string atomic_dir = atomic_path;
      auto pos = atomic_dir.find_last_of("/\\");
      if (pos != std::string::npos) atomic_dir = atomic_dir.substr(0, pos);
      else atomic_dir = ".";
      std::string cmd = "cd " + atomic_dir +
                        " && ./atomic --Z 2 --lmax 0 --mmax 0 --nelem 20 --nnodes 15 --primbas 4"
                        " --method HF --nela 1 --nelb 1 --Rmax 20 --grid 4 --maxit 30"
                        " --save helfem_hf.chk > " + outfn + " 2>&1";
      int rc = std::system(cmd.c_str());
      if (rc != 0) {
        std::cerr << "atomic call failed, return code " << rc << std::endl;
        return 2;
      }
    } else {
      if (!file_exists_local("helfem_hf.chk")) {
        std::cerr << "No local helfem_hf.chk found; set HAVE_ATOMIC to run atomic or provide checkpoint." << std::endl;
        return 2;
      }
    }

    Checkpoint chk("helfem_hf.chk", false);
    helfem::atomic::basis::TwoDBasis basis;
    chk.read(basis);
    basis.compute_tei(true);

    arma::mat Ca;
    chk.read("Ca", Ca);
    arma::uword Norb = std::min<arma::uword>(2, Ca.n_cols);
    if (Norb == 0) {
      std::cerr << "No orbitals in checkpoint." << std::endl;
      return 2;
    }
    arma::mat C_AO = Ca.cols(0, Norb - 1);

    arma::vec nocc(2 * Norb);
    nocc.fill(0.8);

    const double power = 1.0;

    arma::mat Hcore = basis.kinetic() + basis.nuclear();
    arma::mat gC_core, gC_h, gC_xc;
    helfem::rdmft::core_orbital_gradient<helfem::atomic::basis::TwoDBasis>(Hcore, C_AO, nocc, gC_core);
    helfem::rdmft::hartree_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, gC_h);
    helfem::rdmft::muller_xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, power, gC_xc);
    arma::mat gC = gC_core + gC_h + gC_xc;

    const double eps = 1e-6;
    arma::uword max_rows = std::min<arma::uword>(5, C_AO.n_rows);
    arma::uword max_cols = std::min<arma::uword>(2, C_AO.n_cols);

    double max_diff = 0.0;
    for (arma::uword i = 0; i < max_rows; ++i) {
      for (arma::uword j = 0; j < max_cols; ++j) {
        arma::mat Cp = C_AO;
        arma::mat Cm = C_AO;
        Cp(i, j) += eps;
        Cm(i, j) -= eps;
        double Ep = total_energy(basis, Cp, nocc, power);
        double Em = total_energy(basis, Cm, nocc, power);
        double fd = (Ep - Em) / (2.0 * eps);
        double diff = std::abs(fd - gC(i, j));
        if (diff > max_diff) max_diff = diff;
      }
    }

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "Max |finite-diff - analytic| = " << max_diff << "\n";
    if (max_diff > 5e-5) {
      std::cerr << "Orbital gradient check failed: max diff=" << max_diff << std::endl;
      return 2;
    }

    std::cout << "Orbital gradient check passed." << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << "exception: " << ex.what() << std::endl;
    return 2;
  }
  return 0;
}
