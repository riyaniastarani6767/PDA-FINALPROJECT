"""
svm_manual.py
=============
Support Vector Machine (SVM) kernel RBF yang dibuat manual dengan NumPy
(tanpa scikit-learn).

Tahapan mengikuti flowchart Training & Testing SVM (Bab III, Gambar 3.9 dan 3.10):

TRAINING (fit)
  1. Label 0/1 diubah menjadi -1/+1   (Malignant = +1, Benign = -1)
  2. Tentukan kernel RBF serta parameter C dan gamma
  3. Hitung nilai kernel K(xi, xj) untuk setiap pasangan data latih
  4. Selesaikan optimasi soft margin dengan algoritma SMO
     (Sequential Minimal Optimization, Platt 1998)
  5. Simpan support vector (alpha > 0) dan bias b  -> model terlatih
  6. (Opsional) latih fungsi sigmoid (Platt scaling) agar SVM bisa
     mengeluarkan probabilitas malignant

TESTING (predict)
  1. Hitung kernel antara data baru dan support vector
  2. Hitung fungsi keputusan p(x) = sum(alpha_i * y_i * K(x_i, x)) + b
  3. Kelas = sign[p(x)]  -> +1 Malignant (1), -1 Benign (0)

CATATAN
  * Data yang masuk ke fit/predict HARUS sudah distandardisasi (z-score)
    oleh data.py (bagian Teman 2). Kelas ini tidak melakukan standardisasi.
  * Antarmuka sama dengan model RF:
        model.fit(X_train, y_train)   # X: (n, 30), y: 0/1
        model.predict(X)              # -> array 0/1
        model.predict_proba(X)        # -> array probabilitas malignant (0-1)
"""

import numpy as np


class SVMManual:
    def __init__(self, C=1.0, gamma="scale", tol=1e-3, eps=1e-3,
                 max_iter=100_000, probability=True, n_folds_proba=5,
                 random_state=42):
        """
        C            : parameter regularisasi (besar toleransi kesalahan klasifikasi)
        gamma        : parameter kernel RBF. "scale" = 1 / (jumlah_fitur * varians X)
        tol          : toleransi pemeriksaan syarat KKT pada SMO
        eps          : batas perubahan alpha minimum pada SMO
        max_iter     : batas jumlah perulangan SMO (pengaman)
        probability  : True -> latih Platt scaling agar predict_proba tersedia
        n_folds_proba: jumlah fold untuk menghitung nilai keputusan out-of-fold
                       yang dipakai melatih sigmoid (agar probabilitas tidak terlalu yakin)
        random_state : seed agar hasil selalu sama
        """
        self.C = float(C)
        self.gamma = gamma
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.probability = probability
        self.n_folds_proba = n_folds_proba
        self.random_state = random_state

    # ------------------------------------------------------------------ kernel
    def _rbf(self, A, B):
        """Kernel RBF: K(a, b) = exp(-gamma * ||a - b||^2), dihitung sekaligus untuk semua pasangan."""
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
        return np.exp(-self.gamma_ * np.maximum(sq, 0.0))

    # ------------------------------------------------------------------ SMO
    def _smo(self, K, y):
        """
        Menyelesaikan optimasi soft margin (bentuk dual) dengan SMO.
        Memakai konvensi Platt: u_i = sum_j alpha_j y_j K_ij - b, error E_i = u_i - y_i.
        Mengembalikan (alpha, b).
        """
        n = len(y)
        C, tol, eps = self.C, self.tol, self.eps
        alpha = np.zeros(n)
        b = 0.0
        E = -y.astype(float)                 # karena awalnya semua alpha = 0 -> u = 0
        rng = np.random.default_rng(self.random_state)

        def take_step(i1, i2):
            nonlocal b
            if i1 == i2:
                return 0
            a1, a2 = alpha[i1], alpha[i2]
            y1, y2 = y[i1], y[i2]
            E1, E2 = E[i1], E[i2]
            s = y1 * y2
            # batas bawah (L) dan atas (H) untuk alpha2 baru
            if y1 != y2:
                L, H = max(0.0, a2 - a1), min(C, C + a2 - a1)
            else:
                L, H = max(0.0, a2 + a1 - C), min(C, a2 + a1)
            if L >= H:
                return 0
            k11, k12, k22 = K[i1, i1], K[i1, i2], K[i2, i2]
            eta = k11 + k22 - 2.0 * k12
            if eta > 1e-12:
                a2n = np.clip(a2 + y2 * (E1 - E2) / eta, L, H)
            else:
                # kasus jarang (data kembar): bandingkan nilai fungsi objektif di L dan H
                f1 = y1 * (E1 + b) - a1 * k11 - s * a2 * k12
                f2 = y2 * (E2 + b) - s * a1 * k12 - a2 * k22
                L1, H1 = a1 + s * (a2 - L), a1 + s * (a2 - H)
                Lobj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + 0.5 * L ** 2 * k22 + s * L * L1 * k12
                Hobj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + 0.5 * H ** 2 * k22 + s * H * H1 * k12
                if Lobj < Hobj - eps:
                    a2n = L
                elif Lobj > Hobj + eps:
                    a2n = H
                else:
                    a2n = a2
            if abs(a2n - a2) < eps * (a2n + a2 + eps):
                return 0
            a1n = a1 + s * (a2 - a2n)
            # perbarui bias b
            b1 = E1 + y1 * (a1n - a1) * k11 + y2 * (a2n - a2) * k12 + b
            b2 = E2 + y1 * (a1n - a1) * k12 + y2 * (a2n - a2) * k22 + b
            if 0 < a1n < C:
                bn = b1
            elif 0 < a2n < C:
                bn = b2
            else:
                bn = 0.5 * (b1 + b2)
            # perbarui cache error semua data sekaligus
            E[:] += y1 * (a1n - a1) * K[i1] + y2 * (a2n - a2) * K[i2] + b - bn
            alpha[i1], alpha[i2], b = a1n, a2n, bn
            return 1

        def examine(i2):
            y2, a2, E2 = y[i2], alpha[i2], E[i2]
            r2 = E2 * y2
            # hanya proses data yang melanggar syarat KKT
            if (r2 < -tol and a2 < C) or (r2 > tol and a2 > 0):
                non_bound = np.where((alpha > 0) & (alpha < C))[0]
                # heuristik 1: pilih pasangan dengan |E1 - E2| terbesar
                if len(non_bound) > 1:
                    i1 = non_bound[np.argmax(np.abs(E[non_bound] - E2))]
                    if take_step(i1, i2):
                        return 1
                # heuristik 2: coba semua data non-bound mulai dari posisi acak
                for i1 in np.roll(non_bound, rng.integers(max(len(non_bound), 1))):
                    if take_step(i1, i2):
                        return 1
                # heuristik 3: coba semua data mulai dari posisi acak
                for i1 in np.roll(np.arange(n), rng.integers(n)):
                    if take_step(i1, i2):
                        return 1
            return 0

        num_changed, examine_all, it = 0, True, 0
        while (num_changed > 0 or examine_all) and it < self.max_iter:
            num_changed = 0
            idx = range(n) if examine_all else np.where((alpha > 0) & (alpha < C))[0]
            for i in idx:
                num_changed += examine(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            it += 1
        self.n_iter_ = it
        return alpha, b

    # ------------------------------------------------------------------ inti
    def _fit_core(self, X, y_pm):
        """Melatih SVM (tanpa probabilitas). y_pm berisi -1/+1."""
        K = self._rbf(X, X)
        alpha, b = self._smo(K, y_pm)
        sv = alpha > 1e-8
        return X[sv], y_pm[sv], alpha[sv], -b      # intercept = -b (konvensi Platt)

    @staticmethod
    def _decision(Kxs, sv_y, sv_alpha, intercept):
        return Kxs @ (sv_alpha * sv_y) + intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        # langkah 1: label 0/1 -> -1/+1
        y_pm = np.where(y == 1, 1, -1)
        # langkah 2: tentukan gamma
        if self.gamma == "scale":
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        else:
            self.gamma_ = float(self.gamma)
        # langkah 3-5: kernel, optimasi SMO, simpan support vector
        self.sv_X_, self.sv_y_, self.sv_alpha_, self.intercept_ = self._fit_core(X, y_pm)
        self.n_support_ = len(self.sv_alpha_)
        self.n_features_in_ = X.shape[1]
        # langkah 6: Platt scaling (probabilitas)
        if self.probability:
            f_oof = self._decision_out_of_fold(X, y_pm)
            self.prob_A_, self.prob_B_ = self._fit_sigmoid(f_oof, y)
        return self

    def decision_function(self, X):
        """Nilai fungsi keputusan p(x). Positif -> Malignant, negatif -> Benign."""
        X = np.asarray(X, dtype=float)
        return self._decision(self._rbf(X, self.sv_X_), self.sv_y_, self.sv_alpha_, self.intercept_)

    def predict(self, X):
        """Kelas = sign[p(x)]: 1 = Malignant, 0 = Benign."""
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        """Probabilitas malignant (0-1) hasil Platt scaling: P = 1 / (1 + exp(A*p(x) + B))."""
        if not self.probability:
            raise ValueError("Model dilatih dengan probability=False.")
        f = self.decision_function(X)
        z = self.prob_A_ * f + self.prob_B_
        return np.where(z >= 0, np.exp(-z) / (1.0 + np.exp(-z)), 1.0 / (1.0 + np.exp(z)))

    # ------------------------------------------------------------------ Platt scaling
    def _decision_out_of_fold(self, X, y_pm):
        """Nilai keputusan tiap data latih dihitung oleh model yang TIDAK melihat data itu (stratified k-fold)."""
        rng = np.random.default_rng(self.random_state)
        fold = np.empty(len(y_pm), dtype=int)
        for kelas in (-1, 1):
            idx = np.where(y_pm == kelas)[0]
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % self.n_folds_proba
        f = np.zeros(len(y_pm))
        for k in range(self.n_folds_proba):
            tr, te = fold != k, fold == k
            sv_X, sv_y, sv_a, ic = self._fit_core(X[tr], y_pm[tr])
            f[te] = self._decision(self._rbf(X[te], sv_X), sv_y, sv_a, ic)
        return f

    @staticmethod
    def _fit_sigmoid(f, y, max_iter=100):
        """Mencari A dan B pada sigmoid Platt dengan metode Newton (Lin, Lin & Weng, 2007)."""
        prior1, prior0 = int((y == 1).sum()), int((y == 0).sum())
        t = np.where(y == 1, (prior1 + 1.0) / (prior1 + 2.0), 1.0 / (prior0 + 2.0))
        A, B = 0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))
        sigma, min_step = 1e-12, 1e-10

        def objective(A, B):
            z = f * A + B
            return np.sum(np.where(z >= 0, t * z + np.log1p(np.exp(-z)), (t - 1) * z + np.log1p(np.exp(z))))

        fval = objective(A, B)
        for _ in range(max_iter):
            z = f * A + B
            p = np.where(z >= 0, np.exp(-z) / (1 + np.exp(-z)), 1 / (1 + np.exp(z)))
            q = 1 - p
            d2 = p * q
            h11, h22, h21 = sigma + np.sum(f * f * d2), sigma + np.sum(d2), np.sum(f * d2)
            d1 = t - p
            g1, g2 = np.sum(f * d1), np.sum(d1)
            if abs(g1) < 1e-5 and abs(g2) < 1e-5:
                break
            det = h11 * h22 - h21 * h21
            dA, dB = -(h22 * g1 - h21 * g2) / det, -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB
            step = 1.0
            while step >= min_step:
                nA, nB = A + step * dA, B + step * dB
                nf = objective(nA, nB)
                if nf < fval + 1e-4 * step * gd:
                    A, B, fval = nA, nB, nf
                    break
                step /= 2.0
            if step < min_step:
                break
        return A, B