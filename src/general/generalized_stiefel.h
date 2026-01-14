#ifndef HELFEM_GENERALIZED_STIEFEL_H
#define HELFEM_GENERALIZED_STIEFEL_H

#include <armadillo>
#include <stdexcept>

namespace helfem {
namespace manifold {

inline arma::mat symm(const arma::mat& A) {
  return 0.5 * (A + A.t());
}

// Generalized Stiefel manifold: C^T S C = I, with symmetric positive-definite S.
// We use the S-metric <U,V> = tr(U^T S V), which is natural for AO overlap.
// With Euclidean gradient G, the (unconstrained) Riemannian gradient is S^{-1}G.
// Tangent projection under S-metric: grad = S^{-1}G - C * sym(C^T G)
inline arma::mat project_tangent(const arma::mat& C, const arma::mat& G, const arma::mat& S) {
  if (S.n_rows != S.n_cols) throw std::logic_error("S must be square");
  if (C.n_rows != S.n_rows) throw std::logic_error("C and S dimension mismatch");
  if (G.n_rows != C.n_rows || G.n_cols != C.n_cols) throw std::logic_error("G and C dimension mismatch");

  arma::mat SinvG = arma::solve(S, G);
  arma::mat CtG = C.t() * G;
  return SinvG - C * symm(CtG);
}

inline double inner_product(const arma::mat& A, const arma::mat& B, const arma::mat& S) {
  // <A,B>_S = tr(A^T S B)
  return arma::accu(A % (S * B));
}

// Retraction via S-orthonormalization using Cholesky.
// Given Y = C + D, set C_new = Y * inv(chol(Y^T S Y)).
inline arma::mat retract_chol(const arma::mat& C, const arma::mat& D, const arma::mat& S) {
  if (S.n_rows != S.n_cols) throw std::logic_error("S must be square");
  if (C.n_rows != S.n_rows) throw std::logic_error("C and S dimension mismatch");
  if (D.n_rows != C.n_rows || D.n_cols != C.n_cols) throw std::logic_error("D and C dimension mismatch");

  arma::mat Y = C + D;
  arma::mat M = Y.t() * (S * Y);

  // Cholesky expects SPD; add tiny jitter if needed.
  arma::mat R;
  bool ok = arma::chol(R, M);
  if (!ok) {
    arma::mat Mj = M + 1e-12 * arma::eye(M.n_rows, M.n_cols);
    ok = arma::chol(R, Mj);
  }
  if (!ok) throw std::runtime_error("Cholesky failed in retract_chol (matrix not SPD)");

  // arma::chol returns upper-triangular R such that R.t()*R = M
  arma::mat Rinvt = arma::inv(arma::trimatu(R));
  return Y * Rinvt;
}

inline double orthonormality_error(const arma::mat& C, const arma::mat& S) {
  arma::mat I = arma::eye(C.n_cols, C.n_cols);
  arma::mat M = C.t() * (S * C) - I;
  return arma::norm(M, "fro");
}

} // namespace manifold
} // namespace helfem

#endif
