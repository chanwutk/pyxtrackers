# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# ============================================================================
# C-level declaration for the SORT 7-dimensional Kalman filter.
#
# This struct mirrors the filterpy KalmanFilter(dim_x=7, dim_z=4) used in
# the SORT reference. The state vector is [x, y, s, r, vx, vy, vs] where
# (x, y) is the bounding box center, s is scale (area), r is aspect ratio,
# and v* are the corresponding velocities.
#
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py (KalmanFilter7x4 class)
# ============================================================================

# Kalman filter struct for 7D state, 4D measurement.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L9-L16 (KalmanFilter7x4.__init__)
cdef struct KalmanFilter:
    double x[7]          # State vector [x, y, s, r, vx, vy, vs]. Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L11 (self.x)
    double P[7][7]       # Error covariance (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L12 (self.P)
    double Q[7][7]       # Process noise covariance (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L13 (self.Q)
    double F[7][7]       # State transition matrix (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L14 (self.F)
    double H[4][7]       # Measurement matrix (4x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L15 (self.H)
    double R[4][4]       # Measurement noise covariance (4x4). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L16 (self.R)

# Initialize all KF matrices to their default values (identity/zeros).
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L10-L16 (KalmanFilter7x4.__init__)
cdef void kf_init(KalmanFilter *kf) noexcept nogil

# Predict step: x = F*x, P = F*P*F^T + Q.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L24-L32 (KalmanFilter7x4.predict)
cdef void kf_predict(KalmanFilter *kf) noexcept nogil

# Update step: standard Kalman update using Joseph form for numerical stability.
# y = z - H*x, S = H*P*H^T + R, K = P*H^T*S^{-1}, x += K*y,
# P = (I - K*H)*P*(I - K*H)^T + K*R*K^T.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/sort/kalman_filter.py#L35-L64 (KalmanFilter7x4.update)
cdef void kf_update(KalmanFilter *kf, double *z) noexcept nogil
