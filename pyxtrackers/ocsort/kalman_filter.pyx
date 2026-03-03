# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.string cimport memset, memcpy
from libc.math cimport fabs, sqrt
from pyxtrackers.ocsort.kalman_filter cimport KalmanFilter
import cython

# Ref: references/sort/kalman_filter.py#L6 — I7 = np.eye(7)
# Global identity matrix used in the Joseph-form covariance update: P = (I-KH)P(I-KH)^T + KRK^T
cdef double I7[49]

# ============================================================
# Matrix helper functions — replace numpy operations (np.dot, .T, +, -, np.eye, np.zeros, np.linalg.inv).
# No direct reference lines; these are low-level replacements for numpy linear algebra.
# ============================================================

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void matmul(double *A, double *B, double *C, int m, int n, int k) noexcept nogil:
    """C = A * B where A is (m x n), B is (n x k)"""
    # Ref: replaces np.dot() calls in references/sort/kalman_filter.py
    cdef int i, j, l
    cdef double val
    for i in range(m):
        for j in range(k):
            val = 0.0
            for l in range(n):
                val += A[i * n + l] * B[l * k + j]
            C[i * k + j] = val

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_transpose(double *A, double *B, int m, int n) noexcept nogil:
    """B = A^T where A is (m x n)"""
    # Ref: replaces .T (transpose) operations in references/sort/kalman_filter.py
    cdef int i, j
    for i in range(m):
        for j in range(n):
            B[j * m + i] = A[i * n + j]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_add(double *A, double *B, double *C, int size) noexcept nogil:
    """C = A + B"""
    # Ref: replaces numpy + operator on matrices
    cdef int i
    for i in range(size):
        C[i] = A[i] + B[i]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_sub(double *A, double *B, double *C, int size) noexcept nogil:
    """C = A - B"""
    # Ref: replaces numpy - operator on matrices
    cdef int i
    for i in range(size):
        C[i] = A[i] - B[i]

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_eye(double *A, int n) noexcept nogil:
    """Set A to identity matrix of size n x n"""
    # Ref: replaces np.eye(n) in references/sort/kalman_filter.py#L3
    memset(A, 0, n * n * sizeof(double))
    cdef int i
    for i in range(n):
        A[i * n + i] = 1.0

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void mat_zeros(double *A, int size) noexcept nogil:
    """Set A to zeros"""
    # Ref: replaces numpy.zeros() in references/sort/kalman_filter.py#L3
    memset(A, 0, size * sizeof(double))

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef int mat_inv_4x4(double *A, double *inv) noexcept nogil:
    """
    Invert 4x4 matrix A using Gauss-Jordan elimination.
    Returns 1 on success, 0 on failure (singular).
    """
    # Ref: replaces np.linalg.inv(S) in references/sort/kalman_filter.py#L49
    cdef double mat[16]
    memcpy(mat, A, 16 * sizeof(double))
    mat_eye(inv, 4)

    cdef int i, j, k, pivot_idx
    cdef double pivot, temp, factor

    for i in range(4):
        pivot = mat[i*4 + i]
        pivot_idx = i
        for j in range(i + 1, 4):
            if fabs(mat[j*4 + i]) > fabs(pivot):
                pivot = mat[j*4 + i]
                pivot_idx = j

        if fabs(pivot) < 1e-9:
            return 0 # Singular

        if pivot_idx != i:
            for k in range(4):
                temp = mat[i*4 + k]
                mat[i*4 + k] = mat[pivot_idx*4 + k]
                mat[pivot_idx*4 + k] = temp

                temp = inv[i*4 + k]
                inv[i*4 + k] = inv[pivot_idx*4 + k]
                inv[pivot_idx*4 + k] = temp

        pivot = mat[i*4 + i]
        for k in range(4):
            mat[i*4 + k] /= pivot
            inv[i*4 + k] /= pivot

        for j in range(4):
            if i != j:
                factor = mat[j*4 + i]
                for k in range(4):
                    mat[j*4 + k] -= factor * mat[i*4 + k]
                    inv[j*4 + k] -= factor * inv[i*4 + k]
    return 1

# Ref: references/sort/kalman_filter.py#L6 — I7 = np.eye(7)
# Initialize global I7
mat_eye(I7, 7)

# Implementation of public functions declared in .pxd

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_init(KalmanFilter *kf) noexcept nogil:
    # Ref: references/sort/kalman_filter.py#L10-L16 (KalmanFilter7x4.__init__)
    # Initialize state vector (7x1)
    # Ref: references/sort/kalman_filter.py#L11 — self.x = zeros((7, 1))
    mat_zeros(kf.x, 7)
    # Ref: references/sort/kalman_filter.py#L12 — self.P = eye(7)
    # Initialize uncertainty covariance (7x7)
    mat_eye(<double*>kf.P, 7)
    # Ref: references/sort/kalman_filter.py#L13 — self.Q = eye(7)
    # Initialize process uncertainty (7x7)
    mat_eye(<double*>kf.Q, 7)
    # Ref: references/sort/kalman_filter.py#L14 — self.F = eye(7)
    # Initialize state transition matrix (7x7)
    mat_eye(<double*>kf.F, 7)
    # Ref: references/sort/kalman_filter.py#L15 — self.H = zeros((4, 7))
    # Initialize measurement function (4x7)
    mat_zeros(<double*>kf.H, 28)
    # Ref: references/sort/kalman_filter.py#L16 — self.R = eye(4)
    # Initialize state uncertainty (4x4)
    mat_eye(<double*>kf.R, 4)
    # OC-SORT specific: initialize freeze/unfreeze state (no direct ref line;
    # conceptually maps to ocsort.py KalmanBoxTracker needing saved state for occlusion handling)
    kf.has_saved = 0
    kf.observed = 0

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_predict(KalmanFilter *kf) noexcept nogil:
    # Ref: references/sort/kalman_filter.py#L24-L32 (KalmanFilter7x4.predict)
    # Also used by references/ocsort/kalmanfilter.py predict method (identical math).

    # Ref: references/sort/kalman_filter.py#L29 — self.x = dot(F, self.x)
    # x = F * x
    cdef double new_x[7]
    matmul(<double*>kf.F, kf.x, new_x, 7, 7, 1)
    memcpy(kf.x, new_x, 7 * sizeof(double))

    # Ref: references/sort/kalman_filter.py#L32 — self.P = dot(dot(F, self.P), F.T) + Q
    # P = F * P * F^T + Q
    cdef double FP[49] # 7x7
    matmul(<double*>kf.F, <double*>kf.P, FP, 7, 7, 7)

    cdef double FT[49]
    mat_transpose(<double*>kf.F, FT, 7, 7)

    cdef double FPFt[49]
    matmul(FP, FT, FPFt, 7, 7, 7)

    mat_add(FPFt, <double*>kf.Q, <double*>kf.P, 49)

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_update(KalmanFilter *kf, double *z) noexcept nogil:
    # Ref: references/sort/kalman_filter.py#L35-L64 (KalmanFilter7x4.update)
    # Extended with NULL-z handling for OC-SORT: when z is NULL, sets observed=0.
    # Ref: references/ocsort/ocsort.py#L137-L138 — self.kf.update(bbox) / self.kf.update(bbox) where bbox is None

    # Handle NULL z (no observation)
    # Ref: references/ocsort/ocsort.py#L138 — self.kf.update(bbox) with bbox=None
    if z is NULL:
        kf.observed = 0
        return

    # Ref: references/sort/kalman_filter.py#L41 — self.y = z - dot(H, self.x)
    # y = z - H * x
    cdef double Hx[4]
    matmul(<double*>kf.H, kf.x, Hx, 4, 7, 1)

    cdef double y[4]
    mat_sub(z, Hx, y, 4)

    # Ref: references/sort/kalman_filter.py#L44 — PHT = dot(self.P, H.T)
    # S = H * P * H^T + R
    cdef double HT[28] # 7x4
    mat_transpose(<double*>kf.H, HT, 4, 7)

    # Ref: references/sort/kalman_filter.py#L44 — PHT = dot(self.P, H.T)
    cdef double PHt[28] # 7x7 * 7x4 -> 7x4
    matmul(<double*>kf.P, HT, PHt, 7, 7, 4)

    # Ref: references/sort/kalman_filter.py#L48 — S = dot(H, PHT) + R
    cdef double HPHt[16] # 4x7 * 7x4 -> 4x4
    matmul(<double*>kf.H, PHt, HPHt, 4, 7, 4)

    cdef double S[16]
    mat_add(HPHt, <double*>kf.R, S, 16)

    # Ref: references/sort/kalman_filter.py#L49 — SI = np.linalg.inv(S)
    # SI = inv(S)
    cdef double SI[16]
    if mat_inv_4x4(S, SI) == 0:
        return

    # Ref: references/sort/kalman_filter.py#L52 — K = dot(PHT, SI)
    # K = P * H^T * SI = PHt * SI
    cdef double K[28] # 7x4 * 4x4 -> 7x4
    matmul(PHt, SI, K, 7, 4, 4)

    # Ref: references/sort/kalman_filter.py#L56 — self.x = self.x + dot(K, self.y)
    # x = x + K * y
    cdef double Ky[7] # 7x4 * 4x1 -> 7x1
    matmul(K, y, Ky, 7, 4, 1)

    cdef double new_x[7]
    mat_add(kf.x, Ky, new_x, 7)
    memcpy(kf.x, new_x, 7 * sizeof(double))

    # Ref: references/sort/kalman_filter.py#L58-L64
    # P = (I-KH)P(I-KH)^T + KRK^T  (Joseph form, numerically stable)
    # Ref: references/sort/kalman_filter.py#L63 — I_KH = I7 - dot(K, H)
    # I_KH = I - K * H
    cdef double KH[49] # 7x4 * 4x7 -> 7x7
    matmul(K, <double*>kf.H, KH, 7, 4, 7)

    # Use global I7
    cdef double I_KH[49]
    mat_sub(I7, KH, I_KH, 49)

    # Ref: references/sort/kalman_filter.py#L64 — self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, R), K.T)
    # Term 1: I_KH * P * I_KH^T
    cdef double I_KH_P[49]
    matmul(I_KH, <double*>kf.P, I_KH_P, 7, 7, 7)

    cdef double I_KH_T[49]
    mat_transpose(I_KH, I_KH_T, 7, 7)

    cdef double Term1[49]
    matmul(I_KH_P, I_KH_T, Term1, 7, 7, 7)

    # Term 2: K * R * K^T
    cdef double KR[28] # 7x4 * 4x4 -> 7x4
    matmul(K, <double*>kf.R, KR, 7, 4, 4)

    cdef double KT[28] # 4x7
    mat_transpose(K, KT, 7, 4)

    cdef double Term2[49] # 7x4 * 4x7 -> 7x7
    matmul(KR, KT, Term2, 7, 4, 7)

    # P = Term1 + Term2
    mat_add(Term1, Term2, <double*>kf.P, 49)

    # Mark as observed (OC-SORT specific; successful update sets observed=1)
    kf.observed = 1

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_freeze(KalmanFilter *kf) noexcept nogil:
    """Save the parameters before non-observation forward"""
    # Ref: references/ocsort/ocsort.py#L148-L149 — conceptually, the tracker freezes state
    # before lost-state predictions. Maps to KalmanBoxTracker logic that saves x and P
    # when the track enters the lost state, so it can be restored later for smoothing.
    memcpy(kf.x_saved, kf.x, 7 * sizeof(double))
    memcpy(<double*>kf.P_saved, <double*>kf.P, 49 * sizeof(double))
    kf.has_saved = 1

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_unfreeze(KalmanFilter *kf, double *history_obs, int history_len, int max_history) noexcept nogil:
    """
    Unfreeze and perform online smoothing.
    history_obs is a flat array of observations, each observation is 4 doubles (x, y, s, r).
    Only valid observations (non-None) are stored, with None represented as all zeros.
    """
    # Ref: references/ocsort/ocsort.py#L110-L133 — the unfreeze + virtual trajectory
    # interpolation logic in KalmanBoxTracker. Finds last two valid observations,
    # linearly interpolates virtual observations between them, then runs predict/update cycles.

    if kf.has_saved == 0:
        return

    # Restore saved state
    # Ref: references/ocsort/ocsort.py#L110-L111 — restore frozen x and P
    memcpy(kf.x, kf.x_saved, 7 * sizeof(double))
    memcpy(<double*>kf.P, <double*>kf.P_saved, 49 * sizeof(double))

    # Find last two valid observations
    # Ref: references/ocsort/ocsort.py#L113-L120 — find index1 and index2 (last two non-None observations)
    cdef int i, j
    cdef int index1 = -1
    cdef int index2 = -1
    cdef int valid_count = 0
    cdef int is_valid

    # Count valid observations and find last two
    for i in range(history_len - 1, -1, -1):
        # Check if observation is valid (not all zeros)
        is_valid = 0
        for j in range(4):
            if fabs(history_obs[i * 4 + j]) > 1e-9:
                is_valid = 1
                break

        if is_valid:
            valid_count += 1
            if index2 == -1:
                index2 = i
            elif index1 == -1:
                index1 = i
                break

    if index1 == -1 or index2 == -1:
        return

    # Declare all variables at the top
    cdef double x1, y1, s1, r1, w1, h1
    cdef double x2, y2, s2, r2, w2, h2
    cdef int time_gap
    cdef double dx, dy, dw, dh
    cdef double x, y, w, h, s, r
    cdef double new_box[4]

    # Ref: references/ocsort/ocsort.py#L122-L125 — extract box1 (index1) and box2 (index2)
    # in [x, y, s, r] format, convert to [cx, cy, w, h] for interpolation
    # Extract box1 and box2
    x1 = history_obs[index1 * 4 + 0]
    y1 = history_obs[index1 * 4 + 1]
    s1 = history_obs[index1 * 4 + 2]
    r1 = history_obs[index1 * 4 + 3]
    w1 = sqrt(s1 * r1)
    h1 = s1 / w1

    x2 = history_obs[index2 * 4 + 0]
    y2 = history_obs[index2 * 4 + 1]
    s2 = history_obs[index2 * 4 + 2]
    r2 = history_obs[index2 * 4 + 3]
    w2 = sqrt(s2 * r2)
    h2 = s2 / w2

    # Ref: references/ocsort/ocsort.py#L126 — time_gap = index2 - index1
    time_gap = index2 - index1
    if time_gap <= 0:
        return

    # Ref: references/ocsort/ocsort.py#L127-L130 — compute per-step deltas for linear interpolation
    dx = (x2 - x1) / time_gap
    dy = (y2 - y1) / time_gap
    dw = (w2 - w1) / time_gap
    dh = (h2 - h1) / time_gap

    # Ref: references/ocsort/ocsort.py#L131-L133 — generate virtual trajectory and run
    # predict/update cycles for each interpolated observation
    # Generate virtual trajectory
    for i in range(time_gap):
        x = x1 + (i + 1) * dx
        y = y1 + (i + 1) * dy
        w = w1 + (i + 1) * dw
        h = h1 + (i + 1) * dh
        s = w * h
        r = w / h

        new_box[0] = x
        new_box[1] = y
        new_box[2] = s
        new_box[3] = r

        # Update with virtual observation
        kf_update(kf, new_box)

        # Predict for next step (except last)
        if i < time_gap - 1:
            kf_predict(kf)

    kf.has_saved = 0
