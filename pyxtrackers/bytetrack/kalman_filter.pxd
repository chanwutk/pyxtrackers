# cython: language_level=3

# ============================================================================
# C-level declaration for the ByteTrack 8-dimensional Kalman filter.
#
# This struct mirrors the KalmanFilter class from the ByteTrack reference.
# The 8D state space is [x, y, a, h, vx, vy, va, vh] where (x,y) is the
# bounding box center, a is aspect ratio (w/h), h is height, and v* are
# their respective velocities. This is a constant-velocity model with
# state-dependent noise.
#
# Ref: references/bytetrack/kalman_filter.py (KalmanFilter class)
# ============================================================================

# Kalman filter struct for 8D state, 4D measurement.
# Ref: references/bytetrack/kalman_filter.py#L23-L53 (KalmanFilter.__init__)
cdef struct KalmanFilter:
    # State vector (8D: x, y, a, h, vx, vy, va, vh).
    # Ref: references/bytetrack/kalman_filter.py#L72-L74 (mean = np.r_[mean_pos, mean_vel])
    double x[8]
    # Error covariance matrix (8x8).
    # Ref: references/bytetrack/kalman_filter.py#L85 (covariance = np.diag(np.square(std)))
    double P[8][8]
    # State transition matrix (8x8): [[I, dt*I], [0, I]].
    # Ref: references/bytetrack/kalman_filter.py#L44-L46 (self._motion_mat)
    double F[8][8]
    # Measurement/observation matrix (4x8): [I, 0].
    # Ref: references/bytetrack/kalman_filter.py#L47 (self._update_mat)
    double H[4][8]
    # Process noise covariance (8x8), rebuilt each predict step.
    # Ref: references/bytetrack/kalman_filter.py#L117 (motion_cov = np.diag(...))
    double Q[8][8]
    # Measurement noise covariance (4x4), rebuilt each update step.
    # Ref: references/bytetrack/kalman_filter.py#L148 (innovation_cov = np.diag(...))
    double R[4][4]
    # Copy of the motion matrix for prediction. Identical to F.
    # Ref: references/bytetrack/kalman_filter.py#L44 (self._motion_mat)
    double motion_mat[8][8]
    # Copy of the observation matrix for update. Identical to H.
    # Ref: references/bytetrack/kalman_filter.py#L47 (self._update_mat)
    double update_mat[4][8]
    # Position uncertainty weight (1/20). Ref: references/bytetrack/kalman_filter.py#L52
    double std_weight_position
    # Velocity uncertainty weight (1/160). Ref: references/bytetrack/kalman_filter.py#L53
    double std_weight_velocity

# Initialize all KF matrices to default values.
# Ref: references/bytetrack/kalman_filter.py#L40-L53 (KalmanFilter.__init__)
cdef void kf_init(KalmanFilter *kf) noexcept nogil

# Predict step: x = F*x, P = F*P*F^T + Q (state-dependent Q).
# Ref: references/bytetrack/kalman_filter.py#L88-L124 (KalmanFilter.predict)
cdef void kf_predict(KalmanFilter *kf) noexcept nogil

# Update step: project to measurement space, compute Kalman gain, update state.
# Ref: references/bytetrack/kalman_filter.py#L194-L226 (KalmanFilter.update)
cdef void kf_update(KalmanFilter *kf, double *measurement) noexcept nogil

# Create track from unassociated measurement. Outputs initial mean (8D) and
# covariance (8x8 flat). Velocities initialized to zero.
# Ref: references/bytetrack/kalman_filter.py#L55-L86 (KalmanFilter.initiate)
cdef void kf_initiate(KalmanFilter *kf, double *measurement, double *mean_out, double *cov_out) noexcept nogil
