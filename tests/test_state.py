"""Tests that don't need real models or API keys — verify wiring."""

from src.core.state import AgentState, Chunk, RawInput, TraceEvent


def test_state_round_trip():
    state = AgentState(
        session_id="test",
        raw_input=RawInput(text="hello"),
    )
    state.text = "hello"
    state.retrieved_chunks.append(
        Chunk(id="c1", text="some text", metadata={"source": "lec1.pdf", "page": 3})
    )
    state.add_trace(TraceEvent(agent_name="dummy", latency_ms=1.0))

    dumped = state.model_dump_json()
    revived = AgentState.model_validate_json(dumped)
    assert revived.session_id == "test"
    assert revived.retrieved_chunks[0].source_label == "lec1.pdf, p.3"
    assert len(revived.trace) == 1


def test_graph_compiles():
    """Just make sure the graph wires without raising."""
    from src.core.graph import build_graph

    graph = build_graph()
    assert graph is not None
