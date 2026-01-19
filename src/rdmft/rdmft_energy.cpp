#include "rdmft_energy.h"
#include <stdexcept>

namespace helfem {
namespace rdmft {

double core_energy(const arma::mat& Hcore,
                   const arma::mat& C_AO,
                   const arma::vec& n) {
  if(C_AO.n_cols == 0) return 0.0;
  arma::uword Norb = C_AO.n_cols;
  arma::vec n_sum;
  if (n.n_elem == Norb) {
    n_sum = n;
  } else if (n.n_elem == 2 * Norb) {
    n_sum = n.head(Norb) + n.tail(Norb);
  } else {
    throw std::logic_error("core_energy: occupation vector size mismatch");
  }
  // P = C * n * C^T
  arma::mat P = C_AO * arma::diagmat(n_sum) * C_AO.t();
  return arma::trace(P * Hcore);
}

} // namespace rdmft
} // namespace helfem
