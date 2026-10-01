"""Analytic fixture: translate moving pixel centres by (-2, -1) onto LF."""

import numpy as np

from virtual_staining.data.alignment import (
    AlignmentResult,
    AlignmentTransform,
    RegistrationAttempt,
    RegistrationBackend,
    RegistrationRuntime,
)

MATRIX = [[1, 0, -2], [0, 1, -1], [0, 0, 1]]


def register(reference, moving, request):
    return AlignmentResult(
        "succeeded",
        "external_translation",
        request,
        AlignmentTransform(moving.geometry, reference.geometry, "affine", np.array(MATRIX)),
        reason="Analytic fixture, not estimated registration or scientific QC",
        attempt=RegistrationAttempt(
            stage="supplied",
            outcome="succeeded",
            runtime=RegistrationRuntime(
                backend="external_translation", backend_version="1", seed=0
            ),
        ),
    )


BACKEND = RegistrationBackend(register, "external_translation", "1", options={"matrix": MATRIX})
