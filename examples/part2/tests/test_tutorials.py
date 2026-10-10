"""Offline correctness checks. No pretrained download or GPU is required."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.part2.forward import Config, Qwen3, generate, probabilities
from examples.part2.runtime import load_pretrained
from examples.part2.scheduler import KVEngine, Scheduler
from examples.part2.speculative import rejection_distribution, speculative_generate


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(123)
    torch.set_num_threads(1)


@pytest.fixture
def model():
    return Qwen3().eval()


@pytest.mark.parametrize("tied", [False, True])
def test_logits_match_transformers(tied):
    # hidden_size != Hq * head_dim catches a common Qwen3 porting mistake.
    config = Config(hidden_size=24, tie_word_embeddings=tied)
    hf_config = Qwen3Config(**asdict(config), attention_dropout=0.0)
    hf_config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(hf_config).eval()
    ours = Qwen3(config).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        actual, _ = ours(ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_causal_mask_does_not_see_future(model):
    with torch.inference_mode():
        first, _ = model(torch.tensor([[1, 2, 3, 4]]))
        second, _ = model(torch.tensor([[1, 2, 55, 56]]))
    torch.testing.assert_close(first[:, :2], second[:, :2])


@pytest.mark.parametrize("chunk_sizes", [[1, 1, 1, 1, 1], [2, 1, 2], [3, 2]])
def test_cached_logits_match_full_prefix(model, chunk_sizes):
    ids = torch.tensor([[2, 5, 6, 7, 8]])
    with torch.inference_mode():
        expected, _ = model(ids)
        cache, chunks, offset = None, [], 0
        for size in chunk_sizes:
            logits, cache = model(ids[:, offset:offset+size], cache=cache, use_cache=True)
            chunks.append(logits)
            offset += size
        torch.testing.assert_close(torch.cat(chunks, 1), expected, atol=2e-6, rtol=2e-5)
        assert cache[0][0].shape == (1, model.config.num_key_value_heads, 5, model.config.head_dim)


@pytest.mark.parametrize("use_cache", [False, True])
def test_generation_eos_zero_length_and_limit(model, use_cache):
    first = generate(model, [1, 2], 1)[0]
    assert generate(model, [1, 2], 10, eos_id=first, use_cache=use_cache) == [first]
    assert generate(model, [1, 2], 0, use_cache=use_cache) == []
    assert len(generate(model, [1, 2], 7, use_cache=use_cache)) == 7
    assert generate(model, [1, 2], 7, use_cache=True) == generate(model, [1, 2], 7)
    with pytest.raises(ValueError):
        generate(model, [1], 256)


def test_nucleus_keeps_cutoff_token():
    logits = torch.tensor([0.6, 0.3, 0.1]).log()
    torch.testing.assert_close(probabilities(logits, top_p=0.8), torch.tensor([2/3, 1/3, 0.0]))
    torch.testing.assert_close(probabilities(logits, top_k=1), torch.tensor([1.0, 0.0, 0.0]))
    assert probabilities(logits, top_k=100).sum().item() == pytest.approx(1.0)


def test_local_pretrained_loader(tmp_path):
    # A local tiny checkpoint validates the optional loader without downloading 0.6B.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    config = Config(tie_word_embeddings=True)
    reference = Qwen3ForCausalLM(Qwen3Config(**asdict(config))).eval()
    reference.save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(tmp_path)
    loaded, tokenizer = load_pretrained(str(tmp_path))
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids)[0], reference(ids).logits, atol=2e-6, rtol=2e-5)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert tokenizer.encode("hello") == [1]


def test_continuous_batch_matches_serial_and_reuses_slot(model):
    engine = KVEngine(model)
    scheduler = Scheduler(engine)
    a = scheduler.submit("A", [1, 2], 2)
    b = scheduler.submit("B", [3, 4, 5], 5)
    scheduler.step()
    c = scheduler.submit("C", [6], 2)
    while scheduler.has_work:
        scheduler.step()
    assert [x[1] for x in engine.calls] == [("A", "B"), ("A", "B"), ("C",), ("B", "C"), ("B",), ("B",)]
    assert engine.calls[1][2] == (2, 1)  # Decode feeds one token per request, not full prefix.
    assert c.first_token_tick < b.finish_tick
    for request in (a, b, c):
        assert request.output == generate(model, request.prompt, request.max_new_tokens)
        assert request.cache is None
    assert scheduler.reserved == 0


def test_scheduler_cancellation_and_zero_output(model):
    scheduler = Scheduler(KVEngine(model), max_running=1)
    a = scheduler.submit("A", [1, 2], 4)
    b = scheduler.submit("B", [3], 4)
    zero = scheduler.submit("zero", [4], 0)
    assert zero.state == "finished" and zero.output == []
    scheduler.step()
    assert a.cache is not None
    assert scheduler.cancel("B")  # waiting request
    assert scheduler.cancel("A")  # running request
    assert not scheduler.cancel("A") and not scheduler.cancel("unknown")
    assert a.cache is None and b.cache is None
    assert scheduler.reserved == 0 and not scheduler.has_work
    assert len(scheduler.engine.calls) == 1


def test_scheduler_budget_progress_and_oversized_rejection(model):
    scheduler = Scheduler(KVEngine(model), token_capacity=8, prefill_budget=3)
    a = scheduler.submit("A", [1, 2, 3], 3)
    b = scheduler.submit("B", [4, 5], 3)
    while scheduler.has_work:
        scheduler.step()
    assert a.finish_tick < b.first_token_tick
    with pytest.raises(ValueError, match="prefill"):
        scheduler.submit("long_prompt", [1] * 4, 1)
    with pytest.raises(ValueError, match="capacity"):
        scheduler.submit("oversized", [1], 8)
    with pytest.raises(ValueError, match="unique"):
        scheduler.submit("A", [1], 1)


def test_scheduler_eos_and_backend_failure(model, monkeypatch):
    scheduler = Scheduler(KVEngine(model))
    eos = generate(model, [1, 2], 1)[0]
    a = scheduler.submit("A", [1, 2], 5, eos_id=eos)
    scheduler.step()
    assert a.finish_reason == "eos" and a.output == [eos] and a.cache is None
    b = scheduler.submit("B", [3], 2)
    def fail(*args):
        raise RuntimeError("test backend failure")
    monkeypatch.setattr(scheduler.engine, "step", fail)
    with pytest.raises(RuntimeError, match="backend failure"):
        scheduler.step()
    assert b.finish_reason == "error" and scheduler.reserved == 0


class MarkovModel(nn.Module):
    """Known exact conditionals, independent of transformer implementation."""
    def __init__(self, transitions):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("table", torch.tensor(transitions, dtype=torch.float32).log())
        self.config = SimpleNamespace(vocab_size=len(transitions), max_position_embeddings=256)

    def forward(self, ids, **kwargs):
        return self.table[ids], None


@pytest.mark.parametrize("steps", [1, 3, 8])
@pytest.mark.parametrize("same_draft", [False, True])
def test_speculative_greedy_matches_target(model, steps, same_draft):
    draft = model if same_draft else Qwen3(Config(num_hidden_layers=1)).eval()
    expected = generate(model, [1, 2], 13)
    actual, stats = speculative_generate(model, draft, [1, 2], 13, steps)
    assert actual == expected
    if same_draft:
        assert stats.accepted == stats.proposed
        assert stats.target_calls == (13 + steps) // (steps + 1)


def test_speculative_rejection_and_eos_boundaries():
    target = MarkovModel([[0, 1], [0, 1]])
    wrong = MarkovModel([[1, 0], [1, 0]])
    # Proposed EOS=0 is rejected; target correction=1 must be emitted.
    output, stats = speculative_generate(target, wrong, [1], 3, 4, eos_id=0)
    assert output == [1, 1, 1] and stats.accepted == 0
    # Accepted EOS stops immediately, without a bonus token.
    assert speculative_generate(target, target, [0], 5, eos_id=1)[0] == [1]
    # A bonus token can also be EOS: draft stops before the target emits it.
    flip = MarkovModel([[0, 1], [1, 0]])
    assert speculative_generate(flip, flip, [0], 5, 1, eos_id=0)[0] == [1, 0]
    result, stats = speculative_generate(target, wrong, [0], 0)
    assert result == [] and stats.target_calls == stats.draft_calls == 0


def test_rejection_identity_with_disjoint_support():
    p, q = torch.tensor([0.0, 0.3, 0.7]), torch.tensor([0.5, 0.5, 0.0])
    accepted_mass = torch.minimum(p, q)
    corrected = accepted_mass + (1 - accepted_mass.sum()) * rejection_distribution(p, q)
    torch.testing.assert_close(corrected, p)


def test_sampled_two_token_distribution_matches_exact_target():
    target = MarkovModel([[0.8, 0.2], [0.3, 0.7]])
    draft = MarkovModel([[0.1, 0.9], [0.9, 0.1]])
    rng = torch.Generator().manual_seed(2026)
    counts = torch.zeros(2, 2)
    for _ in range(6000):
        result, _ = speculative_generate(target, draft, [0], 2, 2, temperature=1.0, generator=rng)
        counts[result[0], result[1]] += 1
    exact = torch.tensor([[0.64, 0.16], [0.06, 0.14]])
    torch.testing.assert_close(counts / counts.sum(), exact, atol=0.025, rtol=0)


def test_sampling_with_filtered_distributions_and_invalid_inputs(model):
    draft = Qwen3(Config(num_hidden_layers=1)).eval()
    # top-k=1 collapses both distributions to point masses; correction still works.
    actual, _ = speculative_generate(model, draft, [1, 2], 8, temperature=1, top_k=1, top_p=0.8)
    assert actual == generate(model, [1, 2], 8)
    for kwargs in ({"draft_steps": 0}, {"temperature": float("nan")}, {"top_p": 0}):
        with pytest.raises(ValueError):
            speculative_generate(model, draft, [1], **kwargs)
    with pytest.raises(ValueError, match="vocabulary"):
        speculative_generate(model, Qwen3(Config(vocab_size=32)), [1])
