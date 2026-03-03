# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of OC-SORT tracker.

OCSort is a cdef class with C-typed vector fields (no Python objects
in hot path). All internal operations use cdef functions, C structs, and
C++ vectors/maps. The class provides a thin Python-callable update()
method for the wrapper and tests.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from numpy cimport ndarray
from libc.math cimport sqrt, fmax, fmin, isnan, fabs
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap

from pyxtrackers.ocsort.kalman_filter cimport KalmanFilter, kf_init, kf_predict, kf_update, kf_freeze, kf_unfreeze
from pyxtrackers.ocsort.association cimport iou_batch, asso_dispatch, linear_assignment, associate


# ============================================================
# Bounding box conversion utilities (all nogil)
# ============================================================

@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void convert_bbox_to_z(double *bbox, double *z) noexcept nogil:
    """Convert bounding box from [x1,y1,x2,y2] to [x,y,s,r] format.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L21-L33 (convert_bbox_to_z)
    """
    cdef double w = bbox[2] - bbox[0]  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L27 — w = bbox[2] - bbox[0]
    cdef double h = bbox[3] - bbox[1]  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L28 — h = bbox[3] - bbox[1]
    z[0] = bbox[0] + w / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L29 — x = bbox[0] + w/2.
    z[1] = bbox[1] + h / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L30 — y = bbox[1] + h/2.
    z[2] = w * h                       # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L31 — s = w * h
    z[3] = w / (h + 1e-6)              # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L32 — r = w / float(h+1e-6), uses 1e-6 epsilon


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void convert_x_to_bbox(double *x, double *bbox) noexcept nogil:
    """Convert bounding box from [x,y,s,r] to [x1,y1,x2,y2] format.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L36-L46 (convert_x_to_bbox)
    """
    cdef double w = sqrt(x[2] * x[3])  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L41 — w = np.sqrt(x[2] * x[3])
    cdef double h = x[2] / w           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L42 — h = x[2] / w
    bbox[0] = x[0] - w / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L44 — x[0]-w/2.
    bbox[1] = x[1] - h / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L44 — x[1]-h/2.
    bbox[2] = x[0] + w / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L44 — x[0]+w/2.
    bbox[3] = x[1] + h / 2.0           # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L44 — x[1]+h/2.


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void speed_direction(double *bbox1, double *bbox2, double *speed) noexcept nogil:
    """Compute speed direction between two bounding boxes.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L49-L54 (speed_direction)
    """
    cdef double cx1 = (bbox1[0] + bbox1[2]) / 2.0  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L50 — cx1
    cdef double cy1 = (bbox1[1] + bbox1[3]) / 2.0  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L50 — cy1
    cdef double cx2 = (bbox2[0] + bbox2[2]) / 2.0  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L51 — cx2
    cdef double cy2 = (bbox2[1] + bbox2[3]) / 2.0  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L51 — cy2
    cdef double dx = cx2 - cx1                     # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L52 — cx2-cx1
    cdef double dy = cy2 - cy1                     # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L52 — cy2-cy1
    cdef double norm = sqrt(dx * dx + dy * dy) + 1e-6  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L53 — norm
    speed[0] = dy / norm  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L54 — speed / norm, dy component
    speed[1] = dx / norm  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L54 — speed / norm, dx component


# ============================================================
# KalmanBoxTracker struct and cdef functions
# ============================================================

# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L57-L103 (KalmanBoxTracker class)
# Includes observations dict, velocity, last_observation, delta_t.
cdef struct KalmanBoxTracker:
    KalmanFilter kf
    int time_since_update  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L87 — self.time_since_update
    int id                 # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L88 — self.id
    int hits               # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L91 — self.hits
    int hit_streak         # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L92 — self.hit_streak
    int age                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L93 — self.age
    int delta_t            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L103 — self.delta_t
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L99 — self.last_observation = np.array([-1, -1, -1, -1, -1])
    # Last observation [x1, y1, x2, y2, score] or [-1,...] placeholder
    double last_observation[5]
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L102 — self.velocity = None
    # Velocity [dy, dx]
    double velocity[2]
    int has_velocity
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L100 — self.observations = dict()
    # Observations indexed by age: obs[age * 5 + i] for i=0..4
    double *observations
    int observations_capacity
    # Python-style dict keys tracking: which ages have observations
    int *obs_ages
    int obs_ages_len
    int obs_ages_capacity


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_init(KalmanBoxTracker *self, double *bbox, int id, int delta_t) noexcept nogil:
    """Initialize tracker using initial bounding box.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L63-L103 (KalmanBoxTracker.__init__)
    """
    cdef int i, j

    # Initialize Kalman Filter
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L69-L71 — self.kf = KalmanFilter(dim_x=7, dim_z=4)
    kf_init(&self.kf)
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L75-L76 — self.kf.F (constant velocity model)
    # Constant velocity model
    self.kf.F[0][4] = 1.0
    self.kf.F[1][5] = 1.0
    self.kf.F[2][6] = 1.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L77-L78 — self.kf.H (observation matrix)
    # Observation matrix
    self.kf.H[0][0] = 1.0
    self.kf.H[1][1] = 1.0
    self.kf.H[2][2] = 1.0
    self.kf.H[3][3] = 1.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L80 — self.kf.R[2:, 2:] *= 10.
    # Adjust covariance matrices
    self.kf.R[2][2] *= 10.0
    self.kf.R[3][3] *= 10.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L81 — self.kf.P[4:, 4:] *= 1000.
    self.kf.P[4][4] *= 1000.0
    self.kf.P[5][5] *= 1000.0
    self.kf.P[6][6] *= 1000.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L82 — self.kf.P *= 10.
    for i in range(7):
        for j in range(7):
            self.kf.P[i][j] *= 10.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L83 — self.kf.Q[-1, -1] *= 0.01
    self.kf.Q[6][6] *= 0.01
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L84 — self.kf.Q[4:, 4:] *= 0.01
    self.kf.Q[4][4] *= 0.01
    self.kf.Q[5][5] *= 0.01
    self.kf.Q[6][6] *= 0.01

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L86 — self.kf.x[:4] = convert_bbox_to_z(bbox)
    # Initialize state with bbox
    convert_bbox_to_z(bbox, self.kf.x)

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L87-L93 — state fields
    # State fields
    self.time_since_update = 0
    self.id = id
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L103 — self.delta_t = delta_t
    self.delta_t = delta_t

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L99 — self.last_observation = np.array([-1, -1, -1, -1, -1])
    # Last observation placeholder
    for i in range(5):
        self.last_observation[i] = -1.0

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L102 — self.velocity = None
    # Velocity
    self.has_velocity = 0
    self.velocity[0] = 0.0
    self.velocity[1] = 0.0

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L100 — self.observations = dict()
    # Observations array (indexed by age, 5 doubles per entry)
    self.observations_capacity = 200
    self.observations = <double *>calloc(self.observations_capacity * 5, sizeof(double))
    # Mark all as invalid (-1)
    for i in range(self.observations_capacity * 5):
        self.observations[i] = 0.0

    # Track which ages have observations
    self.obs_ages_capacity = 200
    self.obs_ages = <int *>malloc(self.obs_ages_capacity * sizeof(int))
    self.obs_ages_len = 0


cdef void KalmanBoxTracker_destroy(KalmanBoxTracker *self) noexcept nogil:
    """Free observations array."""
    if self.observations != NULL:
        free(self.observations)
        self.observations = NULL
    if self.obs_ages != NULL:
        free(self.obs_ages)
        self.obs_ages = NULL


cdef void _ensure_obs_capacity(KalmanBoxTracker *self, int needed_age) noexcept nogil:
    """Grow observations array if needed."""
    cdef int new_cap, i
    cdef double *new_obs
    cdef int *new_ages
    if needed_age >= self.observations_capacity:
        new_cap = self.observations_capacity * 2
        while needed_age >= new_cap:
            new_cap *= 2
        new_obs = <double *>calloc(new_cap * 5, sizeof(double))
        memcpy(new_obs, self.observations, self.observations_capacity * 5 * sizeof(double))
        free(self.observations)
        self.observations = new_obs
        self.observations_capacity = new_cap
    if self.obs_ages_len >= self.obs_ages_capacity:
        new_cap = self.obs_ages_capacity * 2
        new_ages = <int *>malloc(new_cap * sizeof(int))
        memcpy(new_ages, self.obs_ages, self.obs_ages_len * sizeof(int))
        free(self.obs_ages)
        self.obs_ages = new_ages
        self.obs_ages_capacity = new_cap


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_update(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """Update state vector with observed bbox. bbox is [x1,y1,x2,y2,score] or NULL.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L105-L138 (KalmanBoxTracker.update)
    """
    cdef double prev_bbox[4]
    cdef double z[4]
    cdef int found, i, dt, obs_idx
    cdef double sum_val, w, h

    if bbox != NULL:
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L109-L110 — if bbox is not None, check last_observation
        # Velocity estimation from previous observation
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L110 — if self.last_observation.sum() >= 0
        if self.last_observation[0] >= 0:
            found = 0
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L112-L116 — search delta_t steps back for previous observation
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt >= 0:
                    obs_idx = (self.age - dt) * 5
                    if self.age - dt < self.observations_capacity:
                        sum_val = 0.0
                        for j in range(4):
                            sum_val += fabs(self.observations[obs_idx + j])
                        if sum_val > 1e-9:
                            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L115-L116 — previous_box = self.observations[self.age-dt]
                            # Found valid observation, convert from stored bbox to speed dir
                            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L122 — self.velocity = speed_direction(previous_box, bbox)
                            speed_direction(&self.observations[obs_idx], bbox, self.velocity)
                            self.has_velocity = 1
                            found = 1
                            break
            if not found:
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L117-L118 — if previous_box is None: previous_box = self.last_observation
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L122 — self.velocity = speed_direction(previous_box, bbox)
                speed_direction(self.last_observation, bbox, self.velocity)
                self.has_velocity = 1

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L128-L130 — store observation and update last_observation
        # Store observation in array
        _ensure_obs_capacity(self, self.age)
        obs_idx = self.age * 5
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L129 — self.observations[self.age] = bbox
        self.observations[obs_idx + 0] = bbox[0]
        self.observations[obs_idx + 1] = bbox[1]
        self.observations[obs_idx + 2] = bbox[2]
        self.observations[obs_idx + 3] = bbox[3]
        self.observations[obs_idx + 4] = bbox[4]

        # Track which ages have observations
        self.obs_ages[self.obs_ages_len] = self.age
        self.obs_ages_len += 1

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L128 — self.last_observation = bbox
        # Update last observation
        for i in range(5):
            self.last_observation[i] = bbox[i]

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L132-L135 — update counters
        # Update counters
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L136 — self.kf.update(convert_bbox_to_z(bbox))
        # Update Kalman filter
        convert_bbox_to_z(bbox, z)
        kf_update(&self.kf, z)
    else:
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L137-L138 — self.kf.update(bbox) where bbox is None
        # No observation
        kf_update(&self.kf, NULL)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_predict(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """Advance state vector and return predicted bounding box.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L140-L153 (KalmanBoxTracker.predict)
    """
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L144-L145 — clamp negative scale
    if (self.kf.x[6] + self.kf.x[2]) <= 0:
        self.kf.x[6] = 0.0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L147 — self.kf.predict()
    kf_predict(&self.kf)
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L148 — self.age += 1
    self.age += 1
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L149-L150 — reset hit_streak if time_since_update > 0
    if self.time_since_update > 0:
        self.hit_streak = 0
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L151 — self.time_since_update += 1
    self.time_since_update += 1
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L152 — convert_x_to_bbox(self.kf.x)
    convert_x_to_bbox(self.kf.x, bbox)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void KalmanBoxTracker_get_state(KalmanBoxTracker *self, double *bbox) noexcept nogil:
    """Return current bounding box estimate.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L155-L159 (KalmanBoxTracker.get_state)
    """
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L159 — return convert_x_to_bbox(self.kf.x)
    convert_x_to_bbox(self.kf.x, bbox)


@cython.boundscheck(False)  # type: ignore
@cython.wraparound(False)  # type: ignore
@cython.nonecheck(False)  # type: ignore
cdef void k_previous_obs(KalmanBoxTracker *trk, double *result) noexcept nogil:
    """Get previous observation delta_t steps ago. Returns [x1,y1,x2,y2,score] or [-1,...,-1].
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L10-L18 (k_previous_obs function)
    """
    cdef int i, j, dt, obs_idx
    cdef double sum_val
    cdef int max_age_idx

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L11-L12 — if len(observations) == 0: return [-1,...,-1]
    if trk.obs_ages_len == 0:
        for i in range(5):
            result[i] = -1.0
        return

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L13-L16 — search delta_t steps back
    # Try to find observation delta_t steps ago
    for i in range(trk.delta_t):
        dt = trk.delta_t - i
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L15 — if cur_age - dt in observations
        if trk.age - dt >= 0 and trk.age - dt < trk.observations_capacity:
            obs_idx = (trk.age - dt) * 5
            sum_val = 0.0
            for j in range(4):
                sum_val += fabs(trk.observations[obs_idx + j])
            if sum_val > 1e-9:
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L16 — return observations[cur_age-dt]
                for j in range(5):
                    result[j] = trk.observations[obs_idx + j]
                return

    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L17-L18 — fallback: max_age = max(observations.keys()); return observations[max_age]
    # Fallback: use most recent observation
    max_age_idx = trk.obs_ages[trk.obs_ages_len - 1]
    obs_idx = max_age_idx * 5
    for i in range(5):
        result[i] = trk.observations[obs_idx + i]


# ============================================================
# OCSort cdef class
# ============================================================

# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L175-L191 (OCSort class)
cdef class OCSort:
    """OC-SORT tracker with C-typed vector fields."""
    cdef vector[KalmanBoxTracker*] _trackers
    cdef int _frame_count
    cdef int _max_age
    cdef int _min_hits
    cdef double _iou_threshold
    cdef double _det_thresh
    cdef int _delta_t
    cdef double _inertia
    cdef int _use_byte
    cdef int _asso_func_type
    cdef public int track_id_counter

    def __cinit__(self):
        self._frame_count = 0
        self.track_id_counter = 0

    def __init__(self, det_thresh, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2,
                 use_byte=False):
        """Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L176-L191 (OCSort.__init__)"""
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L181 — self.max_age = max_age
        self._max_age = max_age
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L182 — self.min_hits = min_hits
        self._min_hits = min_hits
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L183 — self.iou_threshold = iou_threshold
        self._iou_threshold = iou_threshold
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L186 — self.det_thresh = det_thresh
        self._det_thresh = det_thresh
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L187 — self.delta_t = delta_t
        self._delta_t = delta_t
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L189 — self.inertia = inertia
        self._inertia = inertia
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L190 — self.use_byte = use_byte
        self._use_byte = 1 if use_byte else 0
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L188 — self.asso_func = ASSO_FUNCS[asso_func]
        # Maps asso_func string to enum type (replaces ASSO_FUNCS dict lookup at L168-172)
        if asso_func == "giou":
            self._asso_func_type = 1
        elif asso_func == "diou":
            self._asso_func_type = 2
        elif asso_func == "ciou":
            self._asso_func_type = 3
        elif asso_func == "ct_dist":
            self._asso_func_type = 4
        else:
            self._asso_func_type = 0  # iou

    def __dealloc__(self):
        cdef int i
        cdef long ptr_val
        cdef cppmap[long, int] freed
        for i in range(<int>self._trackers.size()):
            ptr_val = <long>self._trackers[i]
            if freed.count(ptr_val) == 0 and self._trackers[i] != NULL:
                freed[ptr_val] = 1
                KalmanBoxTracker_destroy(self._trackers[i])
                free(self._trackers[i])
        self._trackers.clear()

    @cython.boundscheck(False)  # type: ignore
    @cython.wraparound(False)  # type: ignore
    @cython.nonecheck(False)  # type: ignore
    cdef int _update_c(self, double *dets_all, double *scores_all, int n_all,
                       double *dets_second_arr, double *scores_second_arr, int n_second,
                       double *output):
        """
        C-level update. Receives pre-parsed, pre-scaled detections.
        Returns: number of output tracks written to output buffer.
        Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L193-L325 (OCSort.update method)
        """

        # All cdef declarations at top (Cython requirement)
        cdef int i, j, k, t, d, idx
        cdef int n_trks, n_dets
        cdef double pred_bbox[4]
        cdef double *dets = NULL
        cdef double *dets_sec = NULL
        cdef double *trks = NULL
        cdef int *to_del = NULL
        cdef int n_del = 0
        cdef int has_nan
        cdef double *velocities = NULL
        cdef double *last_boxes = NULL
        cdef double *k_obs = NULL
        cdef int min_dim, n_matches = 0, n_unmatched_d = 0, n_unmatched_t = 0
        cdef int *match_a = NULL
        cdef int *match_b = NULL
        cdef int *unmatched_d = NULL
        cdef int *unmatched_t = NULL
        cdef double det_bbox[5]
        # BYTE association variables
        cdef int n_ut
        cdef double *u_trk_bb = NULL
        cdef double *iou_left = NULL
        cdef double *sec_bb = NULL
        cdef double max_iou
        cdef double *neg_iou = NULL
        cdef int byte_min, n_bm = 0
        cdef int *bm_a = NULL
        cdef int *bm_b = NULL
        cdef int *to_remove_trk = NULL
        cdef int n_remove_trk = 0
        cdef int *new_ut = NULL
        cdef int new_n_ut = 0
        cdef int is_removed
        # OCR re-association variables
        cdef int n_ld, n_lt
        cdef double *left_det_bb = NULL
        cdef double *left_trk_bb = NULL
        cdef double *iou_left2 = NULL
        cdef double max_iou2
        cdef double *neg_iou2 = NULL
        cdef int ocr_min, n_om = 0
        cdef int *om_a = NULL
        cdef int *om_b = NULL
        cdef int *to_rm_d = NULL
        cdef int *to_rm_t2 = NULL
        cdef int n_rm_d = 0, n_rm_t2 = 0
        cdef int *new_ud2 = NULL
        cdef int new_n_ud2 = 0
        cdef int *new_ut2 = NULL
        cdef int new_n_ut2 = 0
        cdef int rm, rm2
        # New tracker + output variables
        cdef KalmanBoxTracker *new_trk = NULL
        cdef int n_output = 0
        cdef double state_bbox[4]
        cdef double obs_sum
        cdef vector[int] dead_indices
        cdef KalmanBoxTracker *trk_ptr = NULL

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L204 — self.frame_count += 1
        self._frame_count += 1

        # Build dets array (Nx5: x1,y1,x2,y2,score)
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L216-L222 — build dets from bboxes and scores, filter by det_thresh
        n_dets = n_all
        dets = <double *>malloc((n_dets + 1) * 5 * sizeof(double))
        for i in range(n_dets):
            for k in range(4):
                dets[i * 5 + k] = dets_all[i * 4 + k]
            dets[i * 5 + 4] = scores_all[i]

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L219-L220 — dets_second for low-score detections
        # Build dets_second_5 array (Mx5)
        dets_sec = <double *>malloc((n_second + 1) * 5 * sizeof(double))
        for i in range(n_second):
            for k in range(4):
                dets_sec[i * 5 + k] = dets_second_arr[i * 4 + k]
            dets_sec[i * 5 + 4] = scores_second_arr[i]

        # ---- Predict all trackers ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L224-L235 — predict all trackers, remove NaN
        n_trks = <int>self._trackers.size()
        trks = <double *>calloc((n_trks + 1) * 5, sizeof(double))
        to_del = <int *>malloc((n_trks + 1) * sizeof(int))

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L228-L232 — predict each tracker, check for NaN
        for t in range(n_trks):
            KalmanBoxTracker_predict(self._trackers[t], pred_bbox)
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L230 — trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            trks[t * 5 + 0] = pred_bbox[0]
            trks[t * 5 + 1] = pred_bbox[1]
            trks[t * 5 + 2] = pred_bbox[2]
            trks[t * 5 + 3] = pred_bbox[3]
            trks[t * 5 + 4] = 0.0
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L231-L232 — if np.any(np.isnan(pos)): to_del.append(t)
            has_nan = 0
            for k in range(4):
                if isnan(pred_bbox[k]):
                    has_nan = 1
                    break
            if has_nan:
                to_del[n_del] = t
                n_del += 1

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L234-L235 — remove NaN trackers in reverse order
        # Remove NaN trackers (in reverse order)
        for i in range(n_del - 1, -1, -1):
            t = to_del[i]
            KalmanBoxTracker_destroy(self._trackers[t])
            free(self._trackers[t])
            self._trackers.erase(self._trackers.begin() + t)
            for j in range(t, <int>self._trackers.size()):
                for k in range(5):
                    trks[j * 5 + k] = trks[(j + 1) * 5 + k]

        n_trks = <int>self._trackers.size()
        free(to_del)

        # ---- Extract velocities, last_boxes, k_observations ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L237-L241
        velocities = <double *>calloc((n_trks + 1) * 2, sizeof(double))
        last_boxes = <double *>malloc((n_trks + 1) * 5 * sizeof(double))
        k_obs = <double *>malloc((n_trks + 1) * 5 * sizeof(double))

        for t in range(n_trks):
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L237-L238 — velocities array
            if self._trackers[t].has_velocity:
                velocities[t * 2 + 0] = self._trackers[t].velocity[0]
                velocities[t * 2 + 1] = self._trackers[t].velocity[1]
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L239 — last_boxes = [trk.last_observation for ...]
            for k in range(5):
                last_boxes[t * 5 + k] = self._trackers[t].last_observation[k]
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L240-L241 — k_observations = [k_previous_obs(...) for ...]
            k_previous_obs(self._trackers[t], &k_obs[t * 5])

        # ---- First round association ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L246-L249 — first round: associate(dets, trks, ...)
        min_dim = n_dets if n_dets < n_trks else n_trks
        match_a = <int *>malloc((min_dim + 1) * sizeof(int))
        match_b = <int *>malloc((min_dim + 1) * sizeof(int))
        unmatched_d = <int *>malloc((n_dets + 1) * sizeof(int))
        unmatched_t = <int *>malloc((n_trks + 1) * sizeof(int))

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L246-L247 — matched, unmatched_dets, unmatched_trks = associate(...)
        associate(dets, n_dets, trks, n_trks,
                    self._iou_threshold, velocities, k_obs, self._inertia,
                    match_a, match_b, &n_matches,
                    unmatched_d, &n_unmatched_d,
                    unmatched_t, &n_unmatched_t)

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L248-L249 — for m in matched: self.trackers[m[1]].update(dets[m[0], :])
        # Process first-round matches
        for i in range(n_matches):
            d = match_a[i]
            t = match_b[i]
            for k in range(5):
                det_bbox[k] = dets[d * 5 + k]
            KalmanBoxTracker_update(self._trackers[t], det_bbox)

        free(match_a)
        free(match_b)

        # ---- BYTE second-round association (low-score dets) ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L254-L273 — BYTE association with low-score dets
        if self._use_byte and n_second > 0 and n_unmatched_t > 0:
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L256 — u_trks = trks[unmatched_trks]
            n_ut = n_unmatched_t
            u_trk_bb = <double *>malloc(n_ut * 4 * sizeof(double))
            for i in range(n_ut):
                t = unmatched_t[i]
                for k in range(4):
                    u_trk_bb[i * 4 + k] = trks[t * 5 + k]

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L257 — iou_left = self.asso_func(dets_second, u_trks)
            iou_left = <double *>malloc(n_second * n_ut * sizeof(double))
            sec_bb = <double *>malloc(n_second * 4 * sizeof(double))
            for i in range(n_second):
                for k in range(4):
                    sec_bb[i * 4 + k] = dets_sec[i * 5 + k]
            asso_dispatch(self._asso_func_type, sec_bb, n_second, u_trk_bb, n_ut, iou_left)
            free(sec_bb)
            free(u_trk_bb)

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L259 — if iou_left.max() > self.iou_threshold
            max_iou = 0.0
            for i in range(n_second * n_ut):
                if iou_left[i] > max_iou:
                    max_iou = iou_left[i]

            if max_iou > self._iou_threshold:
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L265 — matched_indices = linear_assignment(-iou_left)
                neg_iou = <double *>malloc(n_second * n_ut * sizeof(double))
                for i in range(n_second * n_ut):
                    neg_iou[i] = -iou_left[i]
                byte_min = n_second if n_second < n_ut else n_ut
                bm_a = <int *>malloc((byte_min + 1) * sizeof(int))
                bm_b = <int *>malloc((byte_min + 1) * sizeof(int))
                linear_assignment(neg_iou, n_second, n_ut, bm_a, bm_b, &n_bm)
                free(neg_iou)

                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L267-L272 — filter by threshold, update matched trackers
                to_remove_trk = <int *>malloc((n_bm + 1) * sizeof(int))
                n_remove_trk = 0
                for i in range(n_bm):
                    d = bm_a[i]
                    j = bm_b[i]
                    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L269-L270 — if iou_left[m[0], m[1]] < threshold: continue
                    if iou_left[d * n_ut + j] >= self._iou_threshold:
                        t = unmatched_t[j]
                        for k in range(5):
                            det_bbox[k] = dets_sec[d * 5 + k]
                        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L271 — self.trackers[trk_ind].update(dets_second[det_ind, :])
                        KalmanBoxTracker_update(self._trackers[t], det_bbox)
                        to_remove_trk[n_remove_trk] = t
                        n_remove_trk += 1
                free(bm_a)
                free(bm_b)

                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L273 — unmatched_trks = np.setdiff1d(...)
                new_ut = <int *>malloc((n_unmatched_t + 1) * sizeof(int))
                new_n_ut = 0
                for i in range(n_unmatched_t):
                    is_removed = 0
                    for j in range(n_remove_trk):
                        if unmatched_t[i] == to_remove_trk[j]:
                            is_removed = 1
                            break
                    if not is_removed:
                        new_ut[new_n_ut] = unmatched_t[i]
                        new_n_ut += 1
                free(unmatched_t)
                unmatched_t = new_ut
                n_unmatched_t = new_n_ut
                free(to_remove_trk)

            free(iou_left)

        # ---- OCR re-association with last_boxes ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L275-L297 — re-association using last_boxes
        if n_unmatched_d > 0 and n_unmatched_t > 0:
            n_ld = n_unmatched_d
            n_lt = n_unmatched_t
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L276 — left_dets = dets[unmatched_dets]
            left_det_bb = <double *>malloc(n_ld * 4 * sizeof(double))
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L277 — left_trks = last_boxes[unmatched_trks]
            left_trk_bb = <double *>malloc(n_lt * 4 * sizeof(double))
            for i in range(n_ld):
                d = unmatched_d[i]
                for k in range(4):
                    left_det_bb[i * 4 + k] = dets[d * 5 + k]
            for i in range(n_lt):
                t = unmatched_t[i]
                for k in range(4):
                    left_trk_bb[i * 4 + k] = last_boxes[t * 5 + k]

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L278 — iou_left = self.asso_func(left_dets, left_trks)
            iou_left2 = <double *>malloc(n_ld * n_lt * sizeof(double))
            asso_dispatch(self._asso_func_type, left_det_bb, n_ld, left_trk_bb, n_lt, iou_left2)
            free(left_det_bb)
            free(left_trk_bb)

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L280 — if iou_left.max() > self.iou_threshold
            max_iou2 = 0.0
            for i in range(n_ld * n_lt):
                if iou_left2[i] > max_iou2:
                    max_iou2 = iou_left2[i]

            if max_iou2 > self._iou_threshold:
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L286 — rematched_indices = linear_assignment(-iou_left)
                neg_iou2 = <double *>malloc(n_ld * n_lt * sizeof(double))
                for i in range(n_ld * n_lt):
                    neg_iou2[i] = -iou_left2[i]
                ocr_min = n_ld if n_ld < n_lt else n_lt
                om_a = <int *>malloc((ocr_min + 1) * sizeof(int))
                om_b = <int *>malloc((ocr_min + 1) * sizeof(int))
                linear_assignment(neg_iou2, n_ld, n_lt, om_a, om_b, &n_om)
                free(neg_iou2)

                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L289-L295 — filter by threshold, update matched
                to_rm_d = <int *>malloc((n_om + 1) * sizeof(int))
                to_rm_t2 = <int *>malloc((n_om + 1) * sizeof(int))
                n_rm_d = 0
                n_rm_t2 = 0
                for i in range(n_om):
                    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L291 — if iou_left[m[0], m[1]] < threshold: continue
                    if iou_left2[om_a[i] * n_lt + om_b[i]] >= self._iou_threshold:
                        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L290 — det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                        d = unmatched_d[om_a[i]]
                        t = unmatched_t[om_b[i]]
                        for k in range(5):
                            det_bbox[k] = dets[d * 5 + k]
                        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L293 — self.trackers[trk_ind].update(dets[det_ind, :])
                        KalmanBoxTracker_update(self._trackers[t], det_bbox)
                        to_rm_d[n_rm_d] = d
                        n_rm_d += 1
                        to_rm_t2[n_rm_t2] = t
                        n_rm_t2 += 1
                free(om_a)
                free(om_b)

                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L296 — unmatched_dets = np.setdiff1d(...)
                new_ud2 = <int *>malloc((n_unmatched_d + 1) * sizeof(int))
                new_n_ud2 = 0
                for i in range(n_unmatched_d):
                    rm = 0
                    for j in range(n_rm_d):
                        if unmatched_d[i] == to_rm_d[j]:
                            rm = 1
                            break
                    if not rm:
                        new_ud2[new_n_ud2] = unmatched_d[i]
                        new_n_ud2 += 1
                free(unmatched_d)
                unmatched_d = new_ud2
                n_unmatched_d = new_n_ud2

                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L297 — unmatched_trks = np.setdiff1d(...)
                new_ut2 = <int *>malloc((n_unmatched_t + 1) * sizeof(int))
                new_n_ut2 = 0
                for i in range(n_unmatched_t):
                    rm2 = 0
                    for j in range(n_rm_t2):
                        if unmatched_t[i] == to_rm_t2[j]:
                            rm2 = 1
                            break
                    if not rm2:
                        new_ut2[new_n_ut2] = unmatched_t[i]
                        new_n_ut2 += 1
                free(unmatched_t)
                unmatched_t = new_ut2
                n_unmatched_t = new_n_ut2
                free(to_rm_d)
                free(to_rm_t2)

            free(iou_left2)

        # ---- Update unmatched trackers with None ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L299-L300 — for m in unmatched_trks: self.trackers[m].update(None)
        for i in range(n_unmatched_t):
            KalmanBoxTracker_update(self._trackers[unmatched_t[i]], NULL)

        # ---- Create new trackers for unmatched detections ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L302-L305 — for i in unmatched_dets: trk = KalmanBoxTracker(dets[i,:])
        for i in range(n_unmatched_d):
            d = unmatched_d[i]
            new_trk = <KalmanBoxTracker *>malloc(sizeof(KalmanBoxTracker))
            for k in range(4):
                det_bbox[k] = dets[d * 5 + k]
            KalmanBoxTracker_init(new_trk, det_bbox, self.track_id_counter, self._delta_t)
            self.track_id_counter += 1
            self._trackers.push_back(new_trk)

        # ---- Build output and remove dead trackers ----
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L307-L322
        i = <int>self._trackers.size()
        for idx in range(<int>self._trackers.size() - 1, -1, -1):
            trk_ptr = self._trackers[idx]
            i -= 1

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L308-L315 — use last_observation or KF state
            obs_sum = 0.0
            for k in range(5):
                obs_sum += trk_ptr.last_observation[k]
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L308 — if trk.last_observation.sum() < 0: d = trk.get_state()[0]
            if obs_sum < 0:
                KalmanBoxTracker_get_state(trk_ptr, state_bbox)
            else:
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L315 — d = trk.last_observation[:4]
                for k in range(4):
                    state_bbox[k] = trk_ptr.last_observation[k]

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L316 — if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            if trk_ptr.time_since_update < 1 and (trk_ptr.hit_streak >= self._min_hits or self._frame_count <= self._min_hits):
                output[n_output * 5 + 0] = state_bbox[0]
                output[n_output * 5 + 1] = state_bbox[1]
                output[n_output * 5 + 2] = state_bbox[2]
                output[n_output * 5 + 3] = state_bbox[3]
                # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L318 — [trk.id+1] — +1 for MOT IDs (benchmark requires positive)
                output[n_output * 5 + 4] = <double>(trk_ptr.id + 1)
                n_output += 1

            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L321-L322 — if trk.time_since_update > self.max_age: self.trackers.pop(i)
            if trk_ptr.time_since_update > self._max_age:
                dead_indices.push_back(idx)

        # Remove dead trackers
        for i in range(<int>dead_indices.size()):
            idx = dead_indices[i]
            KalmanBoxTracker_destroy(self._trackers[idx])
            free(self._trackers[idx])
            self._trackers.erase(self._trackers.begin() + idx)

        # ---- Free temporary arrays ----
        free(dets)
        free(dets_sec)
        free(trks)
        free(velocities)
        free(last_boxes)
        free(k_obs)
        free(unmatched_d)
        free(unmatched_t)

        return n_output

    def update(self, dets):
        """
        Update tracker with new detections.

        Args:
            dets: Nx5 float64 array [[x1,y1,x2,y2,score], ...],
                  or None/empty for no detections this frame.
                  Bounding boxes must already be in image coordinates
                  (use ``pyxtrackers.utils.scale`` beforehand if needed).

        Returns:
            Array of tracked objects [[x1,y1,x2,y2,track_id], ...]
        """
        if dets is None or (isinstance(dets, np.ndarray) and dets.size == 0):
            return np.empty((0, 5))

        scores_np = np.asarray(dets[:, 4], dtype=np.float64)
        bboxes_np = dets[:, :4]

        # Filter into high and low score
        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L217-L222 — score filtering
        remain_inds = scores_np > self._det_thresh
        inds_low = scores_np > 0.1
        inds_high = scores_np < self._det_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        cdef cnp.ndarray[cnp.float64_t, ndim=2] dets_high = np.ascontiguousarray(bboxes_np[remain_inds], dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] scores_high = np.ascontiguousarray(scores_np[remain_inds], dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] dets_low = np.ascontiguousarray(bboxes_np[inds_second], dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] scores_low = np.ascontiguousarray(scores_np[inds_second], dtype=np.float64)

        cdef int n_high = dets_high.shape[0]
        cdef int n_low = dets_low.shape[0]

        # Allocate output
        cdef int max_output = n_high + <int>self._trackers.size() + 100
        cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.empty((max_output, 5), dtype=np.float64)

        cdef int n_output = self._update_c(
            <double *>dets_high.data if n_high > 0 else NULL,
            <double *>scores_high.data if n_high > 0 else NULL,
            n_high,
            <double *>dets_low.data if n_low > 0 else NULL,
            <double *>scores_low.data if n_low > 0 else NULL,
            n_low,
            <double *>result.data
        )

        if n_output > 0:
            return result[:n_output]
        return np.empty((0, 5))

    @property
    def trackers(self):
        """Return trackers as list of view objects (backward compat)."""
        return [_make_tracker_view(self._trackers[i]) for i in range(<int>self._trackers.size())]


# ============================================================
# Non-owning view for backward compat
# ============================================================

cdef class KalmanBoxTrackerView:
    """Non-owning view of a C-level KalmanBoxTracker."""
    cdef KalmanBoxTracker *_ptr

    @property
    def time_since_update(self):
        return self._ptr.time_since_update

    @property
    def id(self):
        return self._ptr.id

    @property
    def hits(self):
        return self._ptr.hits

    @property
    def hit_streak(self):
        return self._ptr.hit_streak

    @property
    def age(self):
        return self._ptr.age

    @property
    def last_observation(self):
        return np.array([self._ptr.last_observation[i] for i in range(5)])

    @property
    def velocity(self):
        if self._ptr.has_velocity:
            return np.array([self._ptr.velocity[0], self._ptr.velocity[1]])
        return None

    @property
    def observations(self):
        """Return observations as a Python dict for backward compat."""
        result = {}
        cdef int i, age_val, obs_idx
        for i in range(self._ptr.obs_ages_len):
            age_val = self._ptr.obs_ages[i]
            obs_idx = age_val * 5
            result[age_val] = np.array([
                self._ptr.observations[obs_idx + j] for j in range(5)
            ])
        return result

    def predict(self):
        cdef double bbox[4]
        KalmanBoxTracker_predict(self._ptr, bbox)
        return np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

    def get_state(self):
        cdef double bbox[4]
        KalmanBoxTracker_get_state(self._ptr, bbox)
        return np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape((1, 4))


cdef KalmanBoxTrackerView _make_tracker_view(KalmanBoxTracker *ptr):
    cdef KalmanBoxTrackerView v = KalmanBoxTrackerView.__new__(KalmanBoxTrackerView)
    v._ptr = ptr
    return v
