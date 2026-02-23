import numpy as np
import numpy.typing as npt


class BYTETracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        mot20: bool = False,
        frame_rate: int = 30,
    ):
        ...

    def update(
        self,
        dets: npt.NDArray[np.floating] | None,
    ) -> npt.NDArray[np.floating]:
        ...
