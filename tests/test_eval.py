from tunekit.eval import _prompt_messages, _reference, _split_prompt


class FakeTok:
    """Chat template that mirrors the real ChatML shape closely enough for prefix logic."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msgs)
        return s + ("<|assistant|>" if add_generation_prompt else "")


def test_split_prompt_messages_scores_last_assistant_turn():
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    prompt, full = _split_prompt(FakeTok(), row, "messages")
    assert full.startswith(prompt)
    assert full[len(prompt) :] == "a<|end|>"


def test_split_prompt_other_kinds():
    assert _split_prompt(FakeTok(), {"prompt": "p", "completion": "c"}, "prompt_completion") == (
        "p",
        "pc",
    )
    assert _split_prompt(FakeTok(), {"text": "t"}, "text") == ("", "t")


def test_prompt_and_reference():
    row = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    assert _prompt_messages(row, "messages") == row["messages"][:2]
    assert _reference(row, "messages") == "a"
    assert _prompt_messages({"prompt": "p", "completion": "c"}, "prompt_completion") == [
        {"role": "user", "content": "p"}
    ]
    assert _prompt_messages({"text": "t"}, "text") is None


def test_tiny_eval_set_is_flagged(capsys):
    from tunekit.eval import EvalResult, print_result

    r = EvalResult(
        path="p",
        base_model="b",
        kind="messages",
        examples=2,
        scored_tokens=37,
        tuned_loss=2.0,
        tuned_ppl=7.4,
        base_loss=4.9,
        base_ppl=134.3,
    )
    print_result(r)
    assert "too few" in capsys.readouterr().out


def test_adequate_eval_set_is_not_flagged(capsys):
    from tunekit.eval import EvalResult, print_result

    r = EvalResult(
        path="p",
        base_model="b",
        kind="messages",
        examples=50,
        scored_tokens=5000,
        tuned_loss=2.0,
        tuned_ppl=7.4,
        base_loss=4.9,
        base_ppl=134.3,
    )
    print_result(r)
    assert "too few" not in capsys.readouterr().out
