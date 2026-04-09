import torch
from eval.perplexity import evaluate_perplexity_on_tokens

def test_perplexity_on_known_logits():
    vocab_size, seq_len = 100, 10
    logits = torch.full((1, seq_len, vocab_size), -100.0)
    labels = torch.arange(seq_len).unsqueeze(0)
    # evaluate_perplexity_on_tokens shifts: logits[:, :-1] predicts labels[:, 1:]
    # so logits[0, i] should have high score for labels[0, i+1]
    for i in range(seq_len - 1):
        logits[0, i, labels[0, i + 1]] = 100.0
    ppl = evaluate_perplexity_on_tokens(logits, labels)
    assert ppl < 1.1

def test_perplexity_random_higher():
    vocab_size, seq_len = 100, 50
    logits = torch.randn(1, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (1, seq_len))
    ppl = evaluate_perplexity_on_tokens(logits, labels)
    assert ppl > 10.0
