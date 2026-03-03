# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# ============================================================================
# C-level declaration for the OC-SORT 7-dimensional Kalman filter.
#
# This struct extends the SORT Kalman filter with freeze/unfreeze
# functionality for OC-SORT's occlusion handling. When a track is not
# observed, its state is "frozen" (saved) and later "unfrozen" (restored)
# with online smoothing using virtual trajectory interpolation.
#
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py (KalmanFilterNew class)
# The OC-SORT reference uses the same KalmanFilter as SORT for predict/update,
# but the freeze/unfreeze is implemented in ocsort.py KalmanBoxTracker.
# ============================================================================

cdef struct KalmanFilter:
    double x[7]          # State vector [x, y, s, r, vx, vy, vs]. Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L295 (self.x)
    double P[7][7]       # Error covariance (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L296 (self.P)
    double Q[7][7]       # Process noise covariance (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L297 (self.Q)
    double F[7][7]       # State transition matrix (7x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L299 (self.F)
    double H[4][7]       # Measurement matrix (4x7). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L300 (self.H)
    double R[4][4]       # Measurement noise covariance (4x4). Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/kalmanfilter.py#L301 (self.R)
    # --- OC-SORT specific: freeze/unfreeze state for occlusion handling ---
    double x_saved[7]    # Saved state before non-observation forward pass
    double P_saved[7][7] # Saved covariance before non-observation forward pass
    int has_saved         # 1 if state has been saved (frozen), 0 otherwise
    int observed          # 1 if last update had a real observation, 0 if None

# Ref: KalmanFilterNew.__init__ (references/ocsort/kalmanfilter.py#L283-L337)
cdef void kf_init(KalmanFilter *kf) noexcept nogil

# Ref: KalmanFilterNew.predict (references/ocsort/kalmanfilter.py#L339-L379)
cdef void kf_predict(KalmanFilter *kf) noexcept nogil

# Ref: KalmanFilterNew.update (references/ocsort/kalmanfilter.py#L437-L526)
# Accepts NULL z to indicate no observation (sets observed=0)
cdef void kf_update(KalmanFilter *kf, double *z) noexcept nogil

# OC-SORT specific: save state before non-observation prediction steps.
# Called when a track enters the "lost" state to preserve the last
# observed state for later smoothing.
cdef void kf_freeze(KalmanFilter *kf) noexcept nogil

# OC-SORT specific: restore frozen state and apply online smoothing.
# Interpolates virtual observations between the last two real observations
# and re-runs predict/update cycles to smooth the trajectory.
# Ref: The unfreeze + virtual trajectory logic is from ocsort.py
# KalmanBoxTracker, not directly in kalmanfilter.py.
cdef void kf_unfreeze(KalmanFilter *kf, double *history_obs, int history_len, int max_history) noexcept nogil
