# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of Kalman Filter for ByteTrack.
8-dimensional state space: x, y, a, h, vx, vy, va, vh
"""

cimport cython
from libc.math cimport sqrt
from libc.string cimport memset

# Initialize Kalman Filter with default parameters
@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_init(KalmanFilter *kf) noexcept nogil:
    """
    Initialize Kalman filter with 8-dimensional state space.
    State: [x, y, a, h, vx, vy, va, vh]
    where (x,y) is center, a is aspect ratio, h is height, and v* are velocities.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L40-L53 (KalmanFilter.__init__)
    """
    cdef int i, j
    cdef double dt = 1.0

    # Zero out all matrices
    memset(kf.x, 0, sizeof(double) * 8)
    memset(kf.P, 0, sizeof(double) * 64)
    memset(kf.F, 0, sizeof(double) * 64)
    memset(kf.H, 0, sizeof(double) * 32)
    memset(kf.Q, 0, sizeof(double) * 64)
    memset(kf.R, 0, sizeof(double) * 16)
    memset(kf.motion_mat, 0, sizeof(double) * 64)
    memset(kf.update_mat, 0, sizeof(double) * 32)

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L44-L46 — motion_mat = eye(8) with F[i, ndim+i] = dt
    # Initialize motion matrix (F) - identity with velocity terms
    # F = [[I, dt*I], [0, I]] where I is 4x4 identity
    for i in range(8):
        kf.motion_mat[i][i] = 1.0
    for i in range(4):
        kf.motion_mat[i][4 + i] = dt

    # Copy to F
    for i in range(8):
        for j in range(8):
            kf.F[i][j] = kf.motion_mat[i][j]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L47 — update_mat = eye(ndim, 2*ndim)
    # Initialize update matrix (H) - observation matrix
    # H = [I, 0] where I is 4x4 identity
    for i in range(4):
        kf.update_mat[i][i] = 1.0
        kf.H[i][i] = 1.0

    # Initialize covariance matrices with identity
    for i in range(8):
        kf.P[i][i] = 1.0

    for i in range(8):
        kf.Q[i][i] = 1.0

    for i in range(4):
        kf.R[i][i] = 1.0

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L52-L53 — std_weight_position = 1/20, std_weight_velocity = 1/160
    # Set standard deviation weights
    kf.std_weight_position = 1.0 / 20.0
    kf.std_weight_velocity = 1.0 / 160.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_initiate(KalmanFilter *kf, double *measurement, double *mean_out, double *cov_out) noexcept nogil:
    """
    Create track from unassociated measurement.

    Args:
        measurement: [x, y, a, h] bounding box coordinates
        mean_out: Output mean vector (8 dimensional)
        cov_out: Output covariance matrix (8x8, stored as flat array)
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L55-L86 (initiate)
    """
    cdef int i, j
    cdef double std[8]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L72-L74 — mean = [measurement, zeros(4)] (pos + zero velocity)
    # Initialize mean: [x, y, a, h, 0, 0, 0, 0]
    for i in range(4):
        mean_out[i] = measurement[i]
    for i in range(4, 8):
        mean_out[i] = 0.0

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L76-L84 — std = [2*pos*h, 2*pos*h, 1e-2, 2*pos*h, 10*vel*h, ...]
    # Initialize covariance based on measurement uncertainty
    # std = [2*pos*h, 2*pos*h, 1e-2, 2*pos*h, 10*vel*h, 10*vel*h, 1e-5, 10*vel*h]
    std[0] = 2.0 * kf.std_weight_position * measurement[3]
    std[1] = 2.0 * kf.std_weight_position * measurement[3]
    std[2] = 1e-2
    std[3] = 2.0 * kf.std_weight_position * measurement[3]
    std[4] = 10.0 * kf.std_weight_velocity * measurement[3]
    std[5] = 10.0 * kf.std_weight_velocity * measurement[3]
    std[6] = 1e-5
    std[7] = 10.0 * kf.std_weight_velocity * measurement[3]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L85 — covariance = diag(square(std))
    # Covariance = diag(std^2)
    for i in range(8):
        for j in range(8):
            if i == j:
                cov_out[i * 8 + j] = std[i] * std[i]
            else:
                cov_out[i * 8 + j] = 0.0


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_predict(KalmanFilter *kf) noexcept nogil:
    """
    Run Kalman filter prediction step.
    Updates kf.x and kf.P in place.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L88-L124 (predict)
    """
    cdef int i, j, k
    cdef double new_x[8]
    cdef double new_P[8][8]
    cdef double temp[8][8]
    cdef double std_pos[4]
    cdef double std_vel[4]
    cdef double motion_cov[8][8]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L120 — mean = dot(mean, motion_mat.T)  [equivalent to F * x]
    # Predict mean: x = F * x
    for i in range(8):
        new_x[i] = 0.0
        for j in range(8):
            new_x[i] += kf.F[i][j] * kf.x[j]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L107-L111 — std_pos = [pos_w*h, pos_w*h, 1e-2, pos_w*h]
    # Compute motion covariance based on current state
    std_pos[0] = kf.std_weight_position * kf.x[3]
    std_pos[1] = kf.std_weight_position * kf.x[3]
    std_pos[2] = 1e-2
    std_pos[3] = kf.std_weight_position * kf.x[3]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L112-L116 — std_vel = [vel_w*h, vel_w*h, 1e-5, vel_w*h]
    std_vel[0] = kf.std_weight_velocity * kf.x[3]
    std_vel[1] = kf.std_weight_velocity * kf.x[3]
    std_vel[2] = 1e-5
    std_vel[3] = kf.std_weight_velocity * kf.x[3]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L117 — motion_cov = diag(square([std_pos, std_vel]))
    # Build motion covariance matrix
    for i in range(8):
        for j in range(8):
            motion_cov[i][j] = 0.0

    for i in range(4):
        motion_cov[i][i] = std_pos[i] * std_pos[i]
        motion_cov[i + 4][i + 4] = std_vel[i] * std_vel[i]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L121-L122 — covariance = F @ P @ F.T + motion_cov
    # Predict covariance: P = F * P * F^T + Q
    # First compute temp = F * P
    for i in range(8):
        for j in range(8):
            temp[i][j] = 0.0
            for k in range(8):
                temp[i][j] += kf.F[i][k] * kf.P[k][j]

    # Then compute new_P = temp * F^T
    for i in range(8):
        for j in range(8):
            new_P[i][j] = 0.0
            for k in range(8):
                new_P[i][j] += temp[i][k] * kf.F[j][k]  # F^T[k][j] = F[j][k]

    # Add motion covariance
    for i in range(8):
        for j in range(8):
            new_P[i][j] += motion_cov[i][j]

    # Copy results back
    for i in range(8):
        kf.x[i] = new_x[i]
        for j in range(8):
            kf.P[i][j] = new_P[i][j]


# Note: The Cython version uses simplified diagonal Kalman gain instead of
# Cholesky solve (L216-220 in reference).
@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void kf_update(KalmanFilter *kf, double *measurement) noexcept nogil:
    """
    Run Kalman filter correction step.

    Args:
        measurement: 4-dimensional measurement [x, y, a, h], or NULL for no update
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L194-L226 (update)
    """
    cdef int i, j, k
    cdef double projected_mean[4]
    cdef double projected_cov[4][4]
    cdef double innovation_cov[4][4]
    cdef double std[4]
    cdef double temp1[4][8]
    cdef double temp2[8][4]
    cdef double innovation[4]
    cdef double kalman_gain[8][4]
    cdef double new_mean[8]
    cdef double new_cov[8][8]
    cdef double temp_cov1[8][4]
    cdef double temp_cov2[4][4]

    if measurement == NULL:
        return

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L214 -> references/bytetrack/kalman_filter.py#L126-L153 (project): projected_mean, projected_cov = self.project(mean, cov)
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L150 — projected_mean = H @ mean
    # Project state to measurement space: y = H * x
    for i in range(4):
        projected_mean[i] = 0.0
        for j in range(8):
            projected_mean[i] += kf.H[i][j] * kf.x[j]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L143-L148 — innovation_cov std = [pos_w*h, pos_w*h, 1e-1, pos_w*h]
    # Compute innovation covariance based on current height
    std[0] = kf.std_weight_position * kf.x[3]
    std[1] = kf.std_weight_position * kf.x[3]
    std[2] = 1e-1
    std[3] = kf.std_weight_position * kf.x[3]

    for i in range(4):
        for j in range(4):
            innovation_cov[i][j] = 0.0
    for i in range(4):
        innovation_cov[i][i] = std[i] * std[i]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L151-L152 — projected_cov = H @ P @ H.T
    # Compute projected covariance: S = H * P * H^T
    # temp1 = H * P
    for i in range(4):
        for j in range(8):
            temp1[i][j] = 0.0
            for k in range(8):
                temp1[i][j] += kf.H[i][k] * kf.P[k][j]

    # projected_cov = temp1 * H^T
    for i in range(4):
        for j in range(4):
            projected_cov[i][j] = 0.0
            for k in range(8):
                projected_cov[i][j] += temp1[i][k] * kf.H[j][k]  # H^T[k][j] = H[j][k]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L153 — return mean, covariance + innovation_cov
    # Add innovation covariance
    for i in range(4):
        for j in range(4):
            projected_cov[i][j] += innovation_cov[i][j]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L216-L220 — Kalman gain via Cholesky solve in reference;
    # here simplified with diagonal assumption: K = P * H^T / diag(S)
    # Compute Kalman gain: K = P * H^T * S^(-1)
    # For simplicity, we use a pseudo-inverse approximation
    # temp2 = P * H^T
    for i in range(8):
        for j in range(4):
            temp2[i][j] = 0.0
            for k in range(8):
                temp2[i][j] += kf.P[i][k] * kf.H[j][k]  # H^T[k][j] = H[j][k]

    # Solve K * S = temp2 for K (simplified with diagonal assumption)
    for i in range(8):
        for j in range(4):
            kalman_gain[i][j] = temp2[i][j] / (projected_cov[j][j] + 1e-9)

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L221 — innovation = measurement - projected_mean
    # Compute innovation
    for i in range(4):
        innovation[i] = measurement[i] - projected_mean[i]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L223 — new_mean = mean + dot(innovation, kalman_gain.T)
    # Update mean: x = x + K * innovation
    for i in range(8):
        new_mean[i] = kf.x[i]
        for j in range(4):
            new_mean[i] += kalman_gain[i][j] * innovation[j]

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/kalman_filter.py#L224-L225 — new_cov = cov - K @ projected_cov @ K.T
    # Update covariance: P = P - K * S * K^T
    # temp_cov1 = K * S
    for i in range(8):
        for j in range(4):
            temp_cov1[i][j] = 0.0
            for k in range(4):
                temp_cov1[i][j] += kalman_gain[i][k] * projected_cov[k][j]

    # temp_cov2 = temp_cov1 * K^T
    for i in range(8):
        for j in range(8):
            new_cov[i][j] = kf.P[i][j]
            for k in range(4):
                new_cov[i][j] -= temp_cov1[i][k] * kalman_gain[j][k]  # K^T[k][j] = K[j][k]

    # Copy results back
    for i in range(8):
        kf.x[i] = new_mean[i]
        for j in range(8):
            kf.P[i][j] = new_cov[i][j]
