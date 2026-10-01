from virtual_staining.definitions import Definitions
from virtual_staining.methods.builtin import builtin_definitions

from .method import TINY_CONV, TINY_RESIDUAL, Reconstruction
from .metric import SCALED_MAX_ERROR


def definitions() -> Definitions:
    return builtin_definitions().extend(
        methods=[Reconstruction()],
        components=[TINY_CONV, TINY_RESIDUAL],
        metrics=[SCALED_MAX_ERROR],
    )
