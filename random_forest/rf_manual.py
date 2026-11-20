"""
Random Forest manual dengan kriteria Entropy dan Information Gain

Training:
1. Input data latih (X, y) beserta parameter n_estimators, max_depth,
   min_samples_split, dan max_features.
2. Inisialisasi i = 0 dan list trees = [] untuk menyimpan struktur pohon yang dibangun.
3. Selama i < n_estimators:
   a. Bootstrap Sampling: ambil sampel data latih secara acak dengan pengembalian (replace=True).
   b. Inisialisasi pohon dari root dengan kedalaman awal depth = 0.
   c. Periksa kondisi pemberhentian node:
      - depth >= max_depth (jika max_depth ditentukan), atau
      - jumlah sampel < min_samples_split, atau
      - entropy = 0 (seluruh sampel pada node berasal dari kelas yang sama).
      Jika terpenuhi -> buat Leaf yang menyimpan label kelas mayoritas.
   d. Jika belum berhenti -> pilih m fitur secara acak tanpa pengembalian (m = max_features,
      default sqrt(jumlah_fitur) = sqrt(30) = 5 fitur).
   e. Tentukan kandidat threshold dari nilai tengah (midpoint) antara nilai fitur unik
      yang berurutan, lalu hitung Entropy (Persamaan 2.1) dan Information Gain (Persamaan 2.2).
   f. Pilih kombinasi (fitur, threshold) dengan Information Gain tertinggi sebagai best split.
   g. Bagi data menjadi cabang kiri (<= threshold) dan cabang kanan (> threshold),
      tingkatkan depth + 1, lalu bangun node kiri dan kanan secara rekursif.
   h. Simpan struktur pohon ke dalam trees[] dan lanjutkan ke pohon berikutnya (i = i + 1).
4. Hitung feature importance dari rata-rata penurunan Entropy (Information Gain) tertimbang
   jumlah sampel di seluruh pohon.

Testing 
1. Untuk setiap sampel uji x, telusuri setiap pohon ke-j dari root hingga mencapai leaf:
   - Jika x[fitur_node] <= threshold -> telusuri anak kiri, selain itu -> telusuri anak kanan.
   - Simpan label leaf ke dalam votes[].
2. Majority voting: pilih label dengan frekuensi terbanyak pada votes[] sebagai prediksi akhir.
3. Probabilitas kelas dihitung dari proporsi suara masing-masing kelas pada votes[].
"""

from __future__ import annotations
import math
import numpy as np


def hitung_entropy(y: np.ndarray) -> float:
    """
    Menghitung nilai Entropy H(S) pada suatu simpul S (Persamaan 2.1):
        H(S) = - sum_{i=1}^{c} p_i * log2(p_i)
    Rentang nilai 0 (seluruh data satu kelas) hingga 1 (proporsi kelas seimbang).
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / n
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _entropy_biner_vektor(pos: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Menghitung H(S) secara ter-vektorisasi untuk pemisahan biner (kelas 0 dan 1)."""
    p1 = pos / total
    p0 = 1.0 - p1
    h = np.zeros_like(p1, dtype=float)
    m1 = p1 > 0
    m0 = p0 > 0
    h[m1] -= p1[m1] * np.log2(p1[m1])
    h[m0] -= p0[m0] * np.log2(p0[m0])
    return h


def hitung_information_gain(y: np.ndarray, mask_kiri: np.ndarray) -> float:
    """
    Menghitung Information Gain IG(S, A) dari pemisahan simpul S (Persamaan 2.2):
        IG(S, A) = H(S) - (|S_kiri|/|S| * H(S_kiri) + |S_kanan|/|S| * H(S_kanan))
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    y_kiri = y[mask_kiri]
    y_kanan = y[~mask_kiri]
    if len(y_kiri) == 0 or len(y_kanan) == 0:
        return 0.0
    h_awal = hitung_entropy(y)
    h_split = (len(y_kiri) / n) * hitung_entropy(y_kiri) + (len(y_kanan) / n) * hitung_entropy(y_kanan)
    return float(h_awal - h_split)


def _label_mayoritas(y: np.ndarray) -> int:
    """
    Menentukan label mayoritas pada simpul daun (leaf).
    Jika jumlah suara Benign (0) dan Malignant (1) seri, dipilih Malignant (1)
    untuk meminimalkan False Negative sesuai prioritas medis.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    return 1 if n_pos >= n_neg else 0


class RandomForestManual:
    """Implementasi algoritma Random Forest"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: str | int | float | None = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.trees: list[dict] = []

    def _jumlah_fitur_kandidat(self, n_features: int) -> int:
        """Menentukan jumlah fitur acak m pada setiap simpul"""
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(math.isqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(math.log2(n_features)))
            raise ValueError(f"max_features tidak dikenali: {self.max_features}")
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return max(1, min(int(self.max_features), n_features))

    # training
    def fit(self, X, y) -> RandomForestManual:
        """
        Melatih model Random Forest (Flowchart Gambar 3.8):
        membangun n_estimators pohon keputusan dengan Bootstrap Sampling dan
        pemilihan fitur acak di setiap node menggunakan Entropy & Information Gain.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        self.n_features_ = n_features
        self.m_features_ = self._jumlah_fitur_kandidat(n_features)
        self.trees = []

        rng = np.random.default_rng(self.random_state)
        kepentingan_total = np.zeros(n_features, dtype=float)

        # Langkah 2-3: perulangan pembentukan pohon i = 0 .. n_estimators - 1
        for _ in range(self.n_estimators):
            # Langkah 4: Bootstrap Sampling (ambil sampel acak dengan pengembalian)
            idx_boot = rng.integers(0, n_samples, size=n_samples)
            X_boot = X[idx_boot]
            y_boot = y[idx_boot]

            kepentingan_pohon = np.zeros(n_features, dtype=float)
            # Langkah 5-12: bangun pohon secara rekursif dari depth = 0
            pohon = self._bangun_pohon(
                X_boot, y_boot, depth=0, n_total_pohon=n_samples,
                rng=rng, kepentingan_pohon=kepentingan_pohon,
            )
            self.trees.append(pohon)

            total_pohon = kepentingan_pohon.sum()
            if total_pohon > 0:
                kepentingan_total += kepentingan_pohon / total_pohon

        # Normalisasi feature importance di seluruh pohon agar berjumlah 1.0
        if kepentingan_total.sum() > 0:
            self.feature_importances_ = kepentingan_total / kepentingan_total.sum()
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=float)

        return self

    def _bangun_pohon(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        n_total_pohon: int,
        rng: np.random.Generator,
        kepentingan_pohon: np.ndarray,
    ) -> dict:
        """Membangun simpul pohon keputusan secara rekursif"""
        n_samples = len(y)
        entropy_node = hitung_entropy(y)

        # Langkah 6-7: pemeriksaan kondisi pemberhentian node
        berhenti = (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or (entropy_node == 0.0)
        )
        if berhenti:
            return {
                "leaf": True,
                "label": _label_mayoritas(y),
                "n_samples": n_samples,
                "entropy": entropy_node,
            }

        # Langkah 8: pilih m fitur secara acak sebagai kandidat pemisahan
        fitur_kandidat = rng.choice(self.n_features_, size=self.m_features_, replace=False)

        # Langkah 9: cari fitur dan midpoint threshold dengan Information Gain tertinggi
        fitur_terbaik, threshold_terbaik, ig_terbaik = self._cari_split_terbaik(
            X, y, fitur_kandidat, entropy_node
        )

        # Jika tidak ada pemisahan yang memberikan penurunan entropy (IG <= 0), buat Leaf
        if fitur_terbaik is None or ig_terbaik <= 0.0:
            return {
                "leaf": True,
                "label": _label_mayoritas(y),
                "n_samples": n_samples,
                "entropy": entropy_node,
            }

        # Langkah 10: bagi data menjadi kiri (<= threshold) dan kanan (> threshold)
        mask_kiri = X[:, fitur_terbaik] <= threshold_terbaik
        if mask_kiri.all() or (~mask_kiri).all():
            return {
                "leaf": True,
                "label": _label_mayoritas(y),
                "n_samples": n_samples,
                "entropy": entropy_node,
            }

        # Akumulasi penurunan entropy tertimbang jumlah sampel untuk feature importance
        kepentingan_pohon[fitur_terbaik] += (n_samples / n_total_pohon) * ig_terbaik

        # Langkah 11: bangun node kiri dan node kanan secara rekursif dengan depth + 1
        anak_kiri = self._bangun_pohon(
            X[mask_kiri], y[mask_kiri], depth + 1, n_total_pohon, rng, kepentingan_pohon
        )
        anak_kanan = self._bangun_pohon(
            X[~mask_kiri], y[~mask_kiri], depth + 1, n_total_pohon, rng, kepentingan_pohon
        )

        # Langkah 12: susun struktur node
        return {
            "leaf": False,
            "fitur": int(fitur_terbaik),
            "threshold": float(threshold_terbaik),
            "information_gain": float(ig_terbaik),
            "entropy": float(entropy_node),
            "n_samples": int(n_samples),
            "left": anak_kiri,
            "right": anak_kanan,
        }

    def _cari_split_terbaik(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitur_kandidat: np.ndarray,
        entropy_awal: float,
    ) -> tuple[int | None, float | None, float]:
        """
        Mencari kombinasi (fitur, threshold midpoint) terbaik berdasarkan Information Gain
        tertinggi 
        """
        n = len(y)
        total_pos = int(y.sum())
        fitur_terbaik = None
        threshold_terbaik = None
        ig_terbaik = -1.0

        for idx_fitur in fitur_kandidat:
            kolom = X[:, idx_fitur]
            urut = np.argsort(kolom, kind="mergesort")
            x_urut = kolom[urut]
            y_urut = y[urut]

            # Titik perubahan nilai unik berurutan (kandidat midpoint)
            batas = np.flatnonzero(x_urut[:-1] < x_urut[1:])
            if len(batas) == 0:
                continue

            # Jumlah sampel dan kelas positif di cabang kiri & kanan untuk setiap midpoint
            n_kiri = batas + 1
            n_kanan = n - n_kiri
            pos_kiri = np.cumsum(y_urut)[batas]
            pos_kanan = total_pos - pos_kiri

            # Entropy cabang kiri & kanan (Persamaan 2.1)
            h_kiri = _entropy_biner_vektor(pos_kiri, n_kiri)
            h_kanan = _entropy_biner_vektor(pos_kanan, n_kanan)

            # Information Gain
            ig = entropy_awal - ((n_kiri / n) * h_kiri + (n_kanan / n) * h_kanan)

            idx_maks = int(np.argmax(ig))
            if ig[idx_maks] > ig_terbaik:
                ig_terbaik = float(ig[idx_maks])
                pos_split = batas[idx_maks]
                fitur_terbaik = int(idx_fitur)
                threshold_terbaik = float((x_urut[pos_split] + x_urut[pos_split + 1]) / 2.0)

        return fitur_terbaik, threshold_terbaik, ig_terbaik

    #  testing
    @staticmethod
    def _telusuri_pohon(x: np.ndarray, node: dict) -> int:
        """
        Menelusuri satu pohon keputusan dari root hingga mencapai leaf (Gambar 3.9):
        jika x[fitur] <= threshold -> anak kiri, selain itu -> anak kanan.
        """
        while not node["leaf"]:
            if x[node["fitur"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return int(node["label"])

    def _kumpulkan_suara(self, X: np.ndarray) -> np.ndarray:
        """
        Mengumpulkan hasil prediksi (votes) dari seluruh pohon untuk setiap sampel X.
        Mengembalikan matriks votes berukuran (n_samples, n_estimators).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        votes = np.empty((n_samples, len(self.trees)), dtype=int)
        for i in range(n_samples):
            xi = X[i]
            for j, pohon in enumerate(self.trees):
                votes[i, j] = self._telusuri_pohon(xi, pohon)
        return votes

    def predict_proba(self, X) -> np.ndarray:
        """
        Menghitung probabilitas kelas [P(Benign=0), P(Malignant=1)] untuk setiap sampel
        berdasarkan proporsi suara pohon pada votes[]
        """
        votes = self._kumpulkan_suara(X)
        p_malignant = votes.mean(axis=1)
        return np.column_stack([1.0 - p_malignant, p_malignant])

    def predict(self, X) -> np.ndarray:
        """
        Memprediksi label kelas (0 = Benign, 1 = Malignant) menggunakan Majority Voting
        dari seluruh pohon keputusan pada trees[] 
        """
        votes = self._kumpulkan_suara(X)
        n_pohon = votes.shape[1]
        suara_malignant = votes.sum(axis=1)
        # Majority voting: jika suara Malignant >= setengah jumlah pohon -> 1, selain itu -> 0
        return (suara_malignant * 2 >= n_pohon).astype(int)
