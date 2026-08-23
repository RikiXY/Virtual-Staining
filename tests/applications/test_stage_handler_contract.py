from virtual_staining.applications import pipeline


def test_stage_handlers_resolve_from_applications() -> None:
    for stage in pipeline.VALID_STAGES:
        assert getattr(pipeline, stage).__module__.startswith("virtual_staining.applications.")
