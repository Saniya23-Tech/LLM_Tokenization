
import pickle
import regex as re


# ============================================================
# PART 1: BYTE-TO-UNICODE MAPPING
# ============================================================

def bytes_to_unicode():
    """Map all 256 byte values to printable Unicode characters."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def text_to_bytes(text):
    """Convert text into byte-level Unicode characters."""
    return tuple(B2U[b] for b in text.encode("utf-8"))


# ============================================================
# PART 2: GPT-2-STYLE PRE-TOKENIZATION
# ============================================================

PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text):
    """Split text into GPT-2-style pre-tokens."""
    return PATTERN.findall(text)


# ============================================================
# PART 3: APPLY BPE MERGES
# ============================================================

def merge_word(word, pair, new_token):
    """Merge non-overlapping occurrences of a token pair."""
    new_word = []
    i = 0

    while i < len(word):
        if (
            i < len(word) - 1
            and (word[i], word[i + 1]) == pair
        ):
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


def encode(text, merges):
    """Convert text into token strings using learned merges."""
    all_tokens = []

    for word in pretokenize(text):
        tokens = list(text_to_bytes(word))

        # Apply merge rules in their learned order.
        for pair, new_token in merges.items():
            i = 0

            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    tokens[i:i + 2] = [new_token]
                else:
                    i += 1

        all_tokens.extend(tokens)

    return all_tokens


# ============================================================
# PART 4: TOKEN-TO-ID MAPPING
# ============================================================

def build_vocab_map(vocab):
    """Assign a stable integer ID to each vocabulary token."""
    return {
        token: idx
        for idx, token in enumerate(sorted(vocab))
    }


def encode_to_ids(text, merges, vocab_map):
    """Convert text into integer token IDs."""
    tokens = encode(text, merges)
    ids = []

    for token in tokens:
        if token in vocab_map:
            ids.append(vocab_map[token])
        else:
            # Fall back to individual byte-level characters.
            for char in token:
                if char not in vocab_map:
                    raise ValueError(
                        f"Token character missing from vocabulary: {char!r}"
                    )
                ids.append(vocab_map[char])

    return ids


# ============================================================
# PART 5: DECODING
# ============================================================

def decode(token_ids, id_to_token):
    """Convert token IDs back into the original text."""
    try:
        byte_chars = "".join(
            id_to_token[token_id] for token_id in token_ids
        )
        raw_bytes = bytes(U2B[char] for char in byte_chars)
        return raw_bytes.decode("utf-8", errors="replace")

    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid token IDs or tokenizer vocabulary: {error}"
        ) from error


# ============================================================
# PART 6: BPE TOKENIZER CLASS
# ============================================================

class BPETokenizer:
    """Load, encode, tokenize and decode with a trained BPE model."""

    def __init__(self, merges, vocab_map):
        self.merges = merges
        self.vocab_map = vocab_map
        self.id_to_token = {
            token_id: token
            for token, token_id in vocab_map.items()
        }

    @classmethod
    def load(cls, path):
        """Load a previously saved tokenizer artifact."""
        with open(path, "rb") as file:
            data = pickle.load(file)

        return cls(
            merges=data["merges"],
            vocab_map=data["vocab_map"],
        )

    def tokenize(self, text):
        """Text -> readable token strings."""
        return encode(text, self.merges)

    def encode(self, text):
        """Text -> integer token IDs."""
        return encode_to_ids(
            text,
            self.merges,
            self.vocab_map,
        )

    def decode(self, ids):
        """Integer token IDs -> text."""
        return decode(ids, self.id_to_token)

    @property
    def vocab_size(self):
        return len(self.vocab_map)
