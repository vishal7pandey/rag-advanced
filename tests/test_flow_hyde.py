from app.core.flows import hyde as hy
from app.core.types import Chunk, RetrievedDoc


def test_hyde_timings_and_rrf(monkeypatch):
    # stub seed
    monkeypatch.setattr(hy, "_generate_hyde_seed", lambda q, m, o: "seed text")
    # capture calls to retrieve_dense
    calls = []

    def fake_retrieve(query, cfg):
        calls.append(query)
        # return 3 docs with ids tied to query
        docs = [
            RetrievedDoc(
                chunk=Chunk(
                    id=f"{query[:1]}{i}", doc_id="d", ord=i, text=f"{query} ctx {i}", meta={}
                ),
                score=1.0 / (i + 1),
            )
            for i in range(3)
        ]
        return docs, []

    monkeypatch.setattr(hy, "retrieve_dense", fake_retrieve)
    # stub generate_answer to avoid OpenAI
    monkeypatch.setattr(
        hy,
        "generate_answer",
        lambda prompt, model, offline, docs: {
            "answer_md": "ans",
            "citations": [{"marker": 1}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
    )
    flow = hy.HyDEFlow(
        offline=True, gen_model="stub", emb_st=None, emb_oa=None, params=hy.HyDEParams()
    )
    out = flow.run("Q?", session_state={})
    assert "t_hyde_seed" in out.timings
    # ensure both seed and base queries were retrieved
    assert calls[0] != calls[1]
