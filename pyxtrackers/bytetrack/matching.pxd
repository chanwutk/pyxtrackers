# cython: language_level=3

# ============================================================================
# C-level matching function declarations for cimport by bytetrack.pyx.
#
# These are the Cython equivalents of functions in
# references/bytetrack/matching.py.
# ============================================================================

# Compute IOU-based cost matrix (1 - IOU) from flat tlbr arrays.
# Uses PASCAL VOC formula (+1 to dimensions) matching the reference's
# cython_bbox.bbox_overlaps implementation.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/matching.py#L52-L69 (ious) + L72-90 (iou_distance)
cdef void iou_distance(
    double *atlbrs, int N,
    double *btlbrs, int M,
    double *cost_out
) noexcept nogil

# Fuse detection scores into cost matrix in-place.
# Formula: cost = 1 - (1 - cost) * score = 1 - iou_sim * score
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/matching.py#L172-L180 (fuse_score)
cdef void fuse_score(
    double *cost_matrix,
    double *det_scores,
    int N, int M
) noexcept nogil

# Solve linear assignment problem with threshold, returns via pointer params.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/bytetrack/matching.py#L38-L49 (linear_assignment)
cdef void linear_assignment(
    double *cost_matrix, int N, int M,
    double thresh,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_a, int *n_unmatched_a,
    int *unmatched_b, int *n_unmatched_b
) noexcept nogil
