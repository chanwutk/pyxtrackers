# cython: language_level=3

# ============================================================================
# C-level association function declarations for cimport by ocsort.pyx.
#
# These are the Cython equivalents of functions in
# references/ocsort/association.py.
# ============================================================================

# Association function type enum.
# Maps to ASSO_FUNCS dict in references/ocsort/ocsort.py#L168-L172
cdef enum AssoFuncType:
    ASSO_IOU = 0      # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L5-L21 (iou_batch)
    ASSO_GIOU = 1     # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L24-L56 (giou_batch)
    ASSO_DIOU = 2     # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L59-L96 (diou_batch)
    ASSO_CIOU = 3     # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L98-L149 (ciou_batch)
    ASSO_CT_DIST = 4  # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L152-L173 (ct_dist)

# Compute IOU-based distance from flat bbox arrays (no PASCAL VOC +1).
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L5-L21 (iou_batch)
cdef void iou_batch(double *bb1, int N, double *bb2, int M, double *out) noexcept nogil

# Dispatch to the correct distance metric based on enum.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/ocsort.py#L168-L172 (ASSO_FUNCS dict lookup)
cdef void asso_dispatch(int func_type, double *bb1, int N, double *bb2, int M, double *out) noexcept nogil

# Solve linear assignment on a NEGATED cost matrix, returns raw match pairs.
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L189-L197 (linear_assignment)
cdef void linear_assignment(
    double *cost_matrix, int N, int M,
    int *match_a, int *match_b, int *n_matches
) noexcept nogil

# Full first-round association with velocity direction consistency (VDC).
# Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/ocsort/association.py#L244-L300 (associate)
cdef void associate(
    double *dets, int n_dets,
    double *trks, int n_trks,
    double iou_threshold,
    double *velocities,
    double *previous_obs,
    double vdc_weight,
    int *match_a, int *match_b, int *n_matches,
    int *unmatched_dets, int *n_unmatched_dets,
    int *unmatched_trks, int *n_unmatched_trks
) noexcept nogil
