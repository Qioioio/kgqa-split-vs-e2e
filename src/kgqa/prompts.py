"""Prompts. The wording is in configs/prompts.json; every system uses the same system message.

    plain    the question only (the fine-tuned models E2E and SPLIT)
    rag      three retrieved training questions with their full queries, then the question (E2E-RAG)
    fewshot  eight fixed training questions with their skeletons, then the question (Qwen-FS, Sonnet-FS)
    memory   the question, asking for a query with Wikidata IDs (Sonnet-mem)
"""
IM_END = "<|im_end|>"


def messages(kind, question, prompts, examples=None):
    """Chat messages without the answer. `examples` is a list of (question, query or skeleton)."""
    system = {"role": "system", "content": prompts["system"]}
    if kind in ("plain", "memory"):
        if examples:
            raise ValueError(f"a {kind} prompt takes no examples")
        return [system, {"role": "user", "content": prompts[f"{kind}_user"].format(question=question)}]
    if kind in ("rag", "fewshot"):
        if not examples or len(examples) != prompts[f"{kind}_k"]:
            raise ValueError(f"a {kind} prompt needs exactly {prompts[f'{kind}_k']} examples")
        shown = "".join(prompts[f"{kind}_item"].format(i=i + 1, q=q, sparql=s) for i, (q, s) in enumerate(examples))
        return [system, {"role": "user", "content": prompts[f"{kind}_user"].format(examples=shown, question=question)}]
    raise ValueError(f"unknown prompt kind {kind}")


def answer_span(prompt_text, full_text, answer):
    """Character span [start, end) of the training target inside the templated conversation:
    the answer and the <|im_end|> that closes it. Only this span contributes to the loss.

    prompt_text = chat template of the messages without the answer, with the generation prompt;
    full_text   = chat template of all messages, which for Qwen is prompt_text + answer + "<|im_end|>\\n".
    Anything else raises, so that another template cannot silently shift the labels."""
    if not full_text.startswith(prompt_text):
        raise ValueError("the full text does not start with the prompt")
    rest = full_text[len(prompt_text):]
    if not rest.startswith(answer):
        raise ValueError("the answer does not follow the prompt")
    if rest[len(answer):] != IM_END + "\n":
        raise ValueError(f"unexpected text after the answer: {rest[len(answer):]!r}")
    return len(prompt_text), len(prompt_text) + len(answer) + len(IM_END)


def labels_for_span(input_ids, offsets, span):
    """Token labels: the token id inside the span, -100 elsewhere. A token that straddles the
    border of the span raises."""
    start, end = span
    labels = []
    for token, (s, e) in zip(input_ids, offsets, strict=True):
        if (s < start < e) or (s < end < e):
            raise ValueError("a token straddles the border of the answer")
        labels.append(token if (e > s and s >= start and e <= end) else -100)
    return labels
