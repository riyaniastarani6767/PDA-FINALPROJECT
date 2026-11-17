"""
Support Vector Machine manual dengan kernel RBF
(Laporan Subbab 2.1.8, 2.1.8.1, dan 3.4.4.4 – Flowchart Training & Testing SVM).

Training (Gambar 3.10):
1. Input data latih (30 fitur + label).
2. Ubah label: Malignant = +1, Benign = -1.
3-4. Standardisasi z-score dilakukan di preprocessing (mu dan sigma dari data latih),
     dan disimpan bersama model (lihat train.py).
5. Tentukan kernel RBF serta parameter C dan gamma. Default C = 1,0 dan gamma = 'scale',
   yaitu gamma = 1 / (jumlah_fitur x varians data), sama dengan nilai bawaan scikit-learn.
6. Hitung nilai kernel K(xi, xj) = exp(-gamma * ||xi - xj||^2) (Persamaan 2.6).
7. Selesaikan optimasi soft margin (Persamaan 2.4 dan 2.5) dalam bentuk dual:
       min  1/2 sum_i sum_j a_i a_j y_i y_j K(xi, xj) - sum_i a_i
       syarat 0 <= a_i <= C  dan  sum_i a_i y_i = 0
   menggunakan algoritma SMO (Sequential Minimal Optimization).
8. Peroleh support vector (a_i > 0), bobot (a_i * y_i), dan bias b -> hyperplane (Persamaan 2.3).
9. Model SVM terlatih = support vector, bobot, bias b, C, gamma (+ mu, sigma dari scaler).

Testing (Gambar 3.11):
- Hitung K(x, SV) untuk setiap support vector.
- Fungsi keputusan p(x) = sum_i (a_i y_i) K(x_i, x) + b.
- Jika p(x) > 0 -> Malignant (+1), selain itu -> Benign (-1).

Probabilitas malignant untuk aplikasi diperoleh dengan Platt scaling:
P(malignant | x) = 1 / (1 + exp(A * p(x) + B)).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def kernel_rbf(A, B, gamma):
    """K(a, b) = exp(-gamma * ||a - b||^2) untuk setiap pasangan baris A dan B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    jarak2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(jarak2, 0.0))


class SVMManual:
    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_iter=100_000,
                 probability=False, random_state=42):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.probability = probability
        self.random_state = random_state

    # ------------------------------------------------------------------ training
    def fit(self, X, y01):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y01) == 1, 1.0, -1.0)          # langkah 2

        self.gamma_ = (1.0 / (X.shape[1] * X.var())) if self.gamma == "scale" else float(self.gamma)
        K = kernel_rbf(X, X, self.gamma_)                         # langkah 6
        alpha, b, self.n_iter_ = self._smo(K, y)                  # langkah 7

        sv = alpha > 1e-8                                          # langkah 8
        self.support_ = np.nonzero(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = alpha[sv] * y[sv]                        # bobot a_i * y_i
        self.intercept_ = b
        self.n_support_ = int(sv.sum())
        self.n_features_ = X.shape[1]

        if self.probability:
            self._fit_platt(X, np.asarray(y01))
        return self

    def _smo(self, K, y):
        """
        SMO dengan pemilihan pasangan pelanggar maksimum (maximal violating pair).
        G adalah gradien fungsi tujuan dual; iterasi berhenti saat syarat optimal (KKT)
        terpenuhi dengan toleransi tol.
        """
        n = len(y)
        C = self.C
        Q = (y[:, None] * y[None, :]) * K
        alpha = np.zeros(n)
        G = -np.ones(n)
        it = 0
        while it < self.max_iter:
            yG = -y * G
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            if not up.any() or not low.any():
                break
            i = np.flatnonzero(up)[np.argmax(yG[up])]
            j = np.flatnonzero(low)[np.argmin(yG[low])]
            if yG[i] - yG[j] < self.tol:
                break

            ai_lama, aj_lama = alpha[i], alpha[j]
            if y[i] != y[j]:
                quad = max(Q[i, i] + Q[j, j] + 2.0 * Q[i, j], 1e-12)
                delta = (-G[i] - G[j]) / quad
                diff = alpha[i] - alpha[j]
                alpha[i] += delta
                alpha[j] += delta
                if diff > 0:
                    if alpha[j] < 0:
                        alpha[j], alpha[i] = 0.0, diff
                elif alpha[i] < 0:
                    alpha[i], alpha[j] = 0.0, -diff
                if diff > 0:
                    if alpha[i] > C:
                        alpha[i], alpha[j] = C, C - diff
                elif alpha[j] > C:
                    alpha[j], alpha[i] = C, C + diff
            else:
                quad = max(Q[i, i] + Q[j, j] - 2.0 * Q[i, j], 1e-12)
                delta = (G[i] - G[j]) / quad
                total = alpha[i] + alpha[j]
                alpha[i] -= delta
                alpha[j] += delta
                if total > C:
                    if alpha[i] > C:
                        alpha[i], alpha[j] = C, total - C
                elif alpha[j] < 0:
                    alpha[j], alpha[i] = 0.0, total
                if total > C:
                    if alpha[j] > C:
                        alpha[j], alpha[i] = C, total - C
                elif alpha[i] < 0:
                    alpha[i], alpha[j] = 0.0, total

            G += Q[:, i] * (alpha[i] - ai_lama) + Q[:, j] * (alpha[j] - aj_lama)
            it += 1

        # Bias b dari support vector bebas (0 < a < C)
        yG = -y * G
        bebas = (alpha > 1e-8) & (alpha < C - 1e-8)
        if bebas.any():
            b = float(yG[bebas].mean())
        else:
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            b = float((yG[up].max() + yG[low].min()) / 2.0)
        return alpha, b, it

    # ------------------------------------------------------------------ testing
    def decision_function(self, X):
        """p(x) = sum_i (a_i y_i) K(x_i, x) + b."""
        Kx = kernel_rbf(X, self.support_vectors_, self.gamma_)
        return Kx @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        """p(x) > 0 -> Malignant (1), selain itu Benign (0)."""
        return (self.decision_function(X) > 0).astype(int)

    # -------------------------------------------------------- probabilitas (Platt)
    def _fit_platt(self, X, y01, n_fold=5):
        """
        Parameter A dan B sigmoid diestimasi dari nilai p(x) out-of-fold
        (5-fold stratified pada data latih) agar probabilitas tidak terlalu yakin.
        """
        from preprocessing.data import stratified_kfold

        f = np.zeros(len(y01))
        for _, _, tr, va in stratified_kfold(y01, n_fold, 1, self.random_state):
            m = SVMManual(C=self.C, gamma=self.gamma_, tol=self.tol, max_iter=self.max_iter)
            m.fit(X[tr], y01[tr])
            f[va] = m.decision_function(X[va])

        n_pos, n_neg = int(y01.sum()), int(len(y01) - y01.sum())
        # Target Platt (dengan koreksi prior)
        t = np.where(y01 == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        def nll(p):
            z = p[0] * f + p[1]
            return np.sum(t * np.logaddexp(0, z) + (1 - t) * np.logaddexp(0, -z))

        hasil = minimize(nll, x0=np.array([0.0, np.log((n_neg + 1.0) / (n_pos + 1.0))]), method="BFGS")
        self.prob_a_, self.prob_b_ = float(hasil.x[0]), float(hasil.x[1])

    def predict_proba(self, X):
        if not hasattr(self, "prob_a_"):
            raise RuntimeError("Model dilatih tanpa probability=True.")
        z = self.prob_a_ * self.decision_function(X) + self.prob_b_
        p1 = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
        return np.column_stack([1.0 - p1, p1])
