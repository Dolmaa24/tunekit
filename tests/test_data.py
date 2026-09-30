import json

import pytest
from datasets import Dataset

from tunekit.config import DataConfig
from tunekit.data import (
    DataError,
    alpaca_to_messages,
    detect_format,
    load_raw,
    normalise,
    prepare,
    sharegpt_to_messages,
    validate_messages,
)


def _cfg(**kw):
    return DataConfig(path="unused", **kw)


# -- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, "messages"),
        ({"conversations": [{"from": "human", "value": "hi"}]}, "sharegpt"),
        ({"instruction": "a", "output": "b"}, "alpaca"),
        ({"instruction": "a", "input": "c", "output": "b"}, "alpaca"),
        ({"question": "a", "answer": "b"}, "alpaca"),
        ({"prompt": "a", "completion": "b"}, "prompt_completion"),
        ({"prompt": "a", "response": "b"}, "alpaca"),
        ({"text": "hello"}, "text"),
    ],
)
def test_detect_format(row, expected):
    assert detect_format(row, _cfg()) == expected


def test_detect_dpo_gives_helpful_error():
    with pytest.raises(DataError, match="preference"):
        detect_format({"prompt": "a", "chosen": "b", "rejected": "c"})


def test_detect_unknown():
    with pytest.raises(DataError, match="Could not detect"):
        detect_format({"foo": 1})


def test_custom_text_field():
    assert detect_format({"body": "x"}, _cfg(text_field="body")) == "text"


# -- converters ------------------------------------------------------------


def test_alpaca_with_input():
    msgs = alpaca_to_messages({"instruction": "Sum", "input": "1+1", "output": "2"})
    assert msgs == [
        {"role": "user", "content": "Sum\n\n1+1"},
        {"role": "assistant", "content": "2"},
    ]


def test_alpaca_system_column():
    msgs = alpaca_to_messages({"system": "be terse", "instruction": "hi", "output": "yo"})
    assert msgs[0] == {"role": "system", "content": "be terse"}


def test_sharegpt_roles():
    msgs = sharegpt_to_messages(
        {
            "conversations": [
                {"from": "system", "value": "s"},
                {"from": "human", "value": "h"},
                {"from": "gpt", "value": "g"},
            ]
        }
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_sharegpt_unknown_role():
    with pytest.raises(DataError, match="Unknown sharegpt speaker"):
        sharegpt_to_messages({"conversations": [{"from": "alien", "value": "x"}]})


def test_validate_messages_rules():
    with pytest.raises(DataError, match="no assistant"):
        validate_messages([{"role": "user", "content": "x"}])
    with pytest.raises(DataError, match="end with an assistant"):
        validate_messages([{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}])
    with pytest.raises(DataError, match="System message must be the first"):
        validate_messages(
            [
                {"role": "user", "content": "y"},
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "x"},
            ]
        )


# -- normalise -------------------------------------------------------------


def test_normalise_messages_flattens_parts_and_adds_system():
    ds = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                ]
            }
        ]
    )
    out, kind, fmt, reasons = normalise(ds, _cfg(system_prompt="SYS"))
    assert kind == "messages" and fmt == "messages" and reasons == {}
    assert out[0]["messages"][0] == {"role": "system", "content": "SYS"}
    assert out[0]["messages"][1] == {"role": "user", "content": "hi"}


def test_normalise_drops_bad_rows_but_keeps_good():
    ds = Dataset.from_list(
        [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": ""},  # missing output -> dropped
        ]
    )
    out, kind, _, reasons = normalise(ds, _cfg())
    assert len(out) == 1 and kind == "messages"
    assert sum(reasons.values()) == 1


def test_normalise_all_bad_raises():
    ds = Dataset.from_list([{"instruction": "c", "output": ""}])
    with pytest.raises(DataError, match="Every row was rejected"):
        normalise(ds, _cfg())


def test_normalise_prompt_completion_custom_columns():
    ds = Dataset.from_list([{"q": "a", "a": "b", "junk": 1}])
    out, kind, _, _ = normalise(
        ds, _cfg(format="prompt_completion", prompt_field="q", completion_field="a")
    )
    assert kind == "prompt_completion"
    assert out.column_names == ["prompt", "completion"]


def test_normalise_text_filters_empty():
    ds = Dataset.from_list([{"text": "a"}, {"text": "  "}])
    out, kind, _, _ = normalise(ds, _cfg())
    assert kind == "text" and len(out) == 1


# -- loading + prepare ----------------------------------------------------


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_raw_jsonl_and_dir(tmp_path):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5)]
    _write_jsonl(tmp_path / "a.jsonl", rows[:3])
    _write_jsonl(tmp_path / "b.jsonl", rows[3:])
    assert len(load_raw(str(tmp_path / "a.jsonl"))) == 3
    assert len(load_raw(str(tmp_path))) == 5


def test_load_raw_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("instruction,output\nhi,yo\nhey,sup\n")
    ds = load_raw(str(p))
    assert len(ds) == 2 and detect_format(ds[0]) == "alpaca"


def test_load_raw_missing():
    with pytest.raises(DataError, match="Could not load"):
        load_raw("definitely/not-a-real-dataset-xyz")


def test_prepare_splits_eval(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)])
    nd = prepare(DataConfig(path=str(p), eval_fraction=0.1))
    assert nd.kind == "messages"
    assert len(nd.train) == 90 and len(nd.eval) == 10


def test_prepare_small_dataset_has_no_eval(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5)])
    nd = prepare(DataConfig(path=str(p)))
    assert nd.eval is None and len(nd.train) == 5


def test_prepare_max_samples(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"text": f"t{i}"} for i in range(50)])
    nd = prepare(DataConfig(path=str(p), max_samples=7, eval_fraction=0))
    assert len(nd.train) == 7


def test_missing_local_file_says_so_not_a_hub_error():
    with pytest.raises(DataError, match="No such file"):
        load_raw("data/train.jsonl")


def test_repo_id_shaped_path_still_tries_the_hub():
    # No data-file extension, so it is treated as a Hub id and reports a Hub failure.
    with pytest.raises(DataError, match="Could not load"):
        load_raw("definitely-not/a-real-dataset-xyz")
