from minicoder.model.tokenizer import ByteBPETokenizer, FIRST_MERGE, EOS


def test_roundtrip_bytes():
    tok = ByteBPETokenizer()
    text = "def add(a, b):\n    return a + b  # héllo ✓"
    ids = tok.encode(text)
    assert ids[0] == 257 and ids[-1] == EOS        # BOS/EOS wrapped
    assert tok.decode(ids) == text


def test_bpe_merges_learned_and_save_load(tmp_path):
    tok = ByteBPETokenizer()
    texts = ["the brain learns and the brain remembers"] * 6
    tok.train(texts, target_vocab=FIRST_MERGE + 12)
    assert tok.vocab_size >= FIRST_MERGE
    ids_before = tok.encode("the brain")
    assert any(i >= FIRST_MERGE for i in ids_before)   # merges actually used
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = ByteBPETokenizer.load(path)
    assert tok2.encode("the brain") == ids_before
    assert tok2.decode(tok2.encode("hello world")) == "hello world"


def test_deterministic_training():
    t1, t2 = ByteBPETokenizer(), ByteBPETokenizer()
    corpus = ["repeat repeat repeat abc abc"]
    t1.train(corpus, target_vocab=FIRST_MERGE + 5)
    t2.train(corpus, target_vocab=FIRST_MERGE + 5)
    assert t1.merges == t2.merges
