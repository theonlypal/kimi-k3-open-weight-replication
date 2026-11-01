"""Native channel boundaries and unmodified UTF-8 payload extraction."""
from __future__ import annotations


def find_sequence(tokens, sequence, start=0):
    for index in range(start, len(tokens) - len(sequence) + 1):
        if tokens[index:index + len(sequence)] == sequence:
            return index
    return None


def native_decode(tokenizer, tokens, family):
    if family == "kimi-k3":
        return tokenizer.decode(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def framing(tokenizer, config):
    model_type = config.model_type
    if model_type == "kimi_k3":
        enc = tokenizer.encode
        return {"family": "kimi-k3", "reasoning_open": [],
                "reasoning_end": enc("<|close|>think<|sep|>"),
                "answer_open": enc("<|open|>response<|sep|>"),
                "answer_close": enc("<|close|>response<|sep|>"),
                "message_close": enc("<|close|>message<|sep|>")}
    if model_type == "qwen3":
        return {"family": "qwen3", "reasoning_open": tokenizer.encode("<think>", add_special_tokens=False),
                "reasoning_end": tokenizer.encode("</think>", add_special_tokens=False),
                "answer_open": [], "answer_close": [], "message_close": []}
    raise ValueError("Unsupported native channel framing; no guessed delimiter")


def first_answer_position(tokens, frame):
    end = find_sequence(tokens, frame["reasoning_end"])
    if end is None:
        return None
    start = end + len(frame["reasoning_end"])
    opening = frame["answer_open"]
    if tokens[start:start + len(opening)] != opening:
        return None
    return start + len(opening)


def split_channels(tokens, frame, eos_ids, finish_reason):
    body = list(tokens)
    if finish_reason == "stop":
        if not body or body[-1] not in eos_ids:
            raise ValueError("Normal stop without a selected native stopping token")
        body.pop()
    elif finish_reason != "length":
        raise ValueError("Invalid termination")
    if any(t in eos_ids for t in body):
        raise ValueError("Generation continued after a native stopping token")
    opening = frame.get("reasoning_open", [])
    if body[:len(opening)] != opening:
        raise ValueError("Missing native reasoning-channel opening")
    prefix = body[:len(opening)]
    end = find_sequence(body, frame["reasoning_end"])
    start = first_answer_position(body, frame)
    if end is None:
        return {"reasoning_ids": body[len(prefix):], "answer_ids": [],
                "first_answer_position": None, "answer_channel_reached": False,
                "structure_ids": prefix}
    reasoning = body[len(prefix):end]
    if start is None:
        suffix = body[end + len(frame["reasoning_end"]):]
        if finish_reason != "length" or frame["answer_open"][:len(suffix)] != suffix:
            raise ValueError("Malformed native response-channel opening")
        return {"reasoning_ids": reasoning, "answer_ids": [],
                "first_answer_position": None, "answer_channel_reached": False,
                "structure_ids": prefix + body[end:]}
    answer_end = len(body)
    if frame["answer_close"]:
        closing = find_sequence(body, frame["answer_close"], start)
        if closing is not None:
            answer_end = closing
            tail = body[closing:]
            expected = frame["answer_close"] + frame["message_close"]
            if tail != expected and not (finish_reason == "length" and expected[:len(tail)] == tail):
                raise ValueError("Unexpected native response suffix")
        elif finish_reason == "stop":
            raise ValueError("Normal stop with an unclosed response channel")
        else:
            # A budget may end midway through the structural closing marker.
            for size in range(min(len(frame["answer_close"]) - 1, len(body) - start), 0, -1):
                if body[-size:] == frame["answer_close"][:size]:
                    answer_end = len(body) - size
                    break
    return {"reasoning_ids": reasoning, "answer_ids": body[start:answer_end],
            "first_answer_position": start, "answer_channel_reached": True,
            "structure_ids": prefix + body[end:start] + body[answer_end:]}
