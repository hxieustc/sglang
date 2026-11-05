"""Opt-in owning-worker constructor selection; no native/GPU initialization here."""
import json
import os


def install_native_progress_delay(delay_us, *, rank, endpoint_name, logger):
    if type(delay_us) is not int or delay_us not in (0, 1000):
        raise ValueError("nixl_prog_thread_delay_us must be integer 0 or 1000")
    from kvcr.runtime import progress
    from ._native_progress import bridge

    config_factory, agent_class, source_proof = bridge.actor_factories(progress, delay_us)

    def agent_factory(*args, **kwargs):
        actual = agent_class(*args, **kwargs)
        handoff = actual._poll_delay_construction
        native = handoff["native_config"]
        if not (
            handoff["constructor_returned"] is True
            and handoff["pid"] == os.getpid()
            and handoff["agent_name"] == endpoint_name
            and native["pthrDelay"] == delay_us
            and native["useProgThread"] is True
            and native["useListenThread"] is True
            and native["lthrDelay"] == 100000
            and native["captureTelemetry"] is True
        ):
            raise RuntimeError("Native progress constructor handoff differs from request")
        record = dict(
            event="kvcr_native_progress_constructor", pid=os.getpid(), rank=rank,
            requested_delay_us=delay_us, handoff=handoff, source=source_proof,
            helper_path=__file__,
        )
        logger.info("KVCR native progress constructor %s", json.dumps(record, sort_keys=True))
        return actual

    progress.nixl_agent_config = config_factory
    progress.nixl_agent = agent_factory
