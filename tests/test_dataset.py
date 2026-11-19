from AI.tokenizer import SimpleTokenizer, build_default_tokenizer


def test_special_tokens_present():
    tok = build_default_tokenizer(max_length=16)
    assert tok.pad_id == 0 and tok.unk_id == 1 and tok.bos_id == 2 and tok.eos_id == 3


def test_encode_fixed_length():
    tok = build_default_tokenizer(max_length=20)
    ids = tok.encode("a dragon flying")
    assert len(ids) == 20
    assert ids[0] == tok.bos_id
    assert tok.eos_id in ids


def test_roundtrip():
    tok = build_default_tokenizer(max_length=20)
    decoded = tok.decode(tok.encode("a dragon flying above snowy mountains"))
    assert "dragon" in decoded and "mountain" in decoded


def test_attention_mask():
    tok = build_default_tokenizer(max_length=10)
    mask = tok.attention_mask(tok.encode("hi"))
    assert len(mask) == 10 and mask[0] == 1 and mask[-1] == 0


def test_save_load(tmp_path):
    tok = SimpleTokenizer(max_length=8)
    tok.build_vocab(["hello world", "foo bar"])
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = SimpleTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("hello") == tok.encode("hello")


def test_unknown_token():
    tok = SimpleTokenizer(max_length=8)
    assert tok.unk_id in tok.encode("totally_unknown_word_xyz")
