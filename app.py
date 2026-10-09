import streamlit as st
from pathlib import Path
from tokenization import BPETokenizer

st.set_page_config(
    page_title="My BPE Tokenizer",
    page_icon="🔡",
    layout="wide",
)

st.title("🔡 My Byte-Level BPE Tokenizer")
st.write(
    "Explore how a custom-trained BPE tokenizer "
    "converts text into tokens and token IDs."
)

TOKENIZER_PATH = (
    Path(__file__).parent
    / "artifacts"
    / "bpe_tokenizer.pkl"
)


@st.cache_resource
def load_tokenizer(path):
    return BPETokenizer.load(str(path))


if not TOKENIZER_PATH.exists():
    st.error("Tokenizer artifact not found.")
    st.info("Check that artifacts/bpe_tokenizer.pkl exists.")
    st.stop()

try:
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    st.success(
        f"Tokenizer loaded | Vocabulary: "
        f"{tokenizer.vocab_size:,} tokens"
    )
except Exception as error:
    st.error(f"Tokenizer loading failed: {error}")
    st.stop()

with st.form("tokenizer_form"):
    user_text = st.text_area(
        "Enter your text",
        value="Hello world!",
        height=140,
        placeholder="Type any text here...",
    )
    submitted = st.form_submit_button("Tokenize text")

if submitted:
    if not user_text:
        st.warning("Please enter some text.")
    else:
        try:
            tokens = tokenizer.tokenize(user_text)
            token_ids = tokenizer.encode(user_text)
            decoded_text = tokenizer.decode(token_ids)

            st.subheader("Tokens")
            st.code(repr(tokens), language="python")

            st.subheader("Token IDs")
            st.code(repr(token_ids), language="python")

            st.subheader("Decoded text")
            st.text(decoded_text)

            st.metric("Number of tokens", len(token_ids))

            if decoded_text == user_text:
                st.success("Round-trip test: PASS ✅")
            else:
                st.warning(
                    "Decoded text differs from the original input."
                )

        except Exception as error:
            st.error(f"Tokenization failed: {error}")