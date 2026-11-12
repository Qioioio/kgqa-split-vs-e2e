import pytest

from kgqa.prompts import IM_END, answer_span, labels_for_span, messages

PROMPTS = {"system": "S", "plain_user": "{question}", "memory_user": "IDs: {question}",
           "rag_k": 2, "rag_user": "{examples}Q: {question}", "rag_item": "{q} -> {sparql}\n",
           "fewshot_k": 1, "fewshot_user": "{examples}Q: {question}", "fewshot_item": "{q} => {sparql}\n"}


def test_messages():
    assert messages("plain", "Who?", PROMPTS) == [{"role": "system", "content": "S"}, {"role": "user", "content": "Who?"}]
    assert messages("rag", "Who?", PROMPTS, [("a", "A"), ("b", "B")])[1]["content"] == "a -> A\nb -> B\nQ: Who?"
    with pytest.raises(ValueError):
        messages("rag", "Who?", PROMPTS, [("a", "A")])           # wrong number of examples
    with pytest.raises(ValueError):
        messages("plain", "Who?", PROMPTS, [("a", "A")])


def test_answer_span_covers_the_answer_and_its_end_token():
    prompt = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
    full = prompt + "SELECT ?x" + IM_END + "\n"
    start, end = answer_span(prompt, full, "SELECT ?x")
    assert full[start:end] == "SELECT ?x" + IM_END
    with pytest.raises(ValueError):
        answer_span(prompt, full + "more", "SELECT ?x")
    with pytest.raises(ValueError):
        answer_span("other", full, "SELECT ?x")


def test_labels_cover_exactly_the_tokens_inside_the_span():
    ids = [10, 11, 12, 13]
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert labels_for_span(ids, offsets, (3, 9)) == [-100, 11, 12, -100]
    assert labels_for_span(ids, offsets, (0, 12)) == ids
    with pytest.raises(ValueError):
        labels_for_span(ids, offsets, (4, 9))                   # a token straddles the border
