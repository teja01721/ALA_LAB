import re
import numpy as np
import gensim.downloader as api

# ==================================================
# Load GloVe Model
# ==================================================

print("Loading GloVe model...")
model = api.load("glove-wiki-gigaword-50")

# ==================================================
# Tags (Exactly 20)
# ==================================================

TAGS = [
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "technology",
    "curriculum",
    "collaboration",
    "publication",
    "laboratory",
    "scholarship",
    "mentorship",
    "internship",
    "entrepreneurship",
    "accreditation",
    "alumni"
]

# ==================================================
# Stopwords
# ==================================================

STOPWORDS = {
    "the","a","an","and","or","of","to",
    "in","on","for","is","are","was","were",
    "with","at","by","from","this","that",
    "it","our","we","be","as","has","have"
}

# ==================================================
# Vector Class
# ==================================================

class Vector:

    def __init__(self, values):
        self.values = np.array(values)

    def dot(self, other):
        return np.sum(self.values * other.values)

    def norm(self):
        return np.sqrt(np.sum(self.values ** 2))

    def cosine_similarity(self, other):
        return self.dot(other) / (self.norm() * other.norm())

# ==================================================
# Text Preprocessing
# ==================================================

def preprocess_text(text):

    total_before = len(text.split())

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]

    total_after = len(tokens)

    return tokens, total_before, total_after

# ==================================================
# Tag Matrix
# ==================================================

def build_tag_matrix(model, tags):

    tag_vectors = []

    for tag in tags:

        if tag not in model.key_to_index:
            raise ValueError(
                f"Tag not found in vocabulary: {tag}"
            )

        tag_vectors.append(model[tag])

    T = np.array(tag_vectors)

    return tags, T

# ==================================================
# Text Matrix
# ==================================================

def build_text_matrix(model, tokens):

    in_vocab_tokens = []
    oov_tokens = []

    for token in tokens:

        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)

    if len(in_vocab_tokens) == 0:
        raise ValueError(
            "No in-vocabulary words found."
        )

    W = np.array(
        [model[word] for word in in_vocab_tokens]
    )

    return in_vocab_tokens, oov_tokens, W

# ==================================================
# Similarity Matrix using Vector Class
# ==================================================

def similarity_matrix_vector_class(W, T):

    n = W.shape[0]
    m = T.shape[0]

    S = np.zeros((n, m))

    for i in range(n):

        for j in range(m):

            word_vector = Vector(W[i])

            tag_vector = Vector(T[j])

            S[i, j] = word_vector.cosine_similarity(
                tag_vector
            )

    return S

# ==================================================
# Similarity Matrix using NumPy
# ==================================================

def similarity_matrix_numpy(W, T):

    W_norm = np.linalg.norm(
        W,
        axis=1,
        keepdims=True
    )

    T_norm = np.linalg.norm(
        T,
        axis=1,
        keepdims=True
    )

    W_hat = W / W_norm

    T_hat = T / T_norm

    S = W_hat @ T_hat.T

    return S

# ==================================================
# Rank Tags
# ==================================================

def rank_tags(tag_names, S, in_vocab_tokens):

    results = []

    for j in range(len(tag_names)):

        idx = np.argmax(S[:, j])

        score = S[idx, j]

        best_word = in_vocab_tokens[idx]

        results.append(
            (
                tag_names[j],
                score,
                best_word
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results

# ==================================================
# Main Program
# ==================================================

def main():

    with open(
        "manipal_text.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    tokens, before, after = preprocess_text(text)

    tag_names, T = build_tag_matrix(
        model,
        TAGS
    )

    in_vocab_tokens, oov_tokens, W = \
        build_text_matrix(
            model,
            tokens
        )

    print("\n===== TOKEN STATISTICS =====")

    print("Tokens before preprocessing:", before)

    print("Tokens after preprocessing:", after)

    print("In-vocabulary tokens:",
          len(in_vocab_tokens))

    print("Out-of-vocabulary tokens:",
          len(oov_tokens))

    print("Distinct OOV tokens:")

    print(set(oov_tokens))

    print("\n===== MATRIX SHAPES =====")

    print("T shape =", T.shape)

    print("W shape =", W.shape)

    S_vector = similarity_matrix_vector_class(
        W,
        T
    )

    S_numpy = similarity_matrix_numpy(
        W,
        T
    )

    print("\n===== VERIFICATION =====")

    print(
        "np.allclose =",
        np.allclose(
            S_vector,
            S_numpy
        )
    )

    max_diff = np.max(
        np.abs(
            S_vector - S_numpy
        )
    )

    print(
        "Maximum difference =",
        max_diff
    )

    ranking = rank_tags(
        tag_names,
        S_numpy,
        in_vocab_tokens
    )

    print("\n===== TOP 8 TAGS =====")

    print(
        f"{'Rank':<6}"
        f"{'Tag':<20}"
        f"{'Score':<12}"
        f"{'Best Word'}"
    )

    for rank, (
        tag,
        score,
        word
    ) in enumerate(
        ranking[:8],
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{tag:<20}"
            f"{score:<12.4f}"
            f"{word}"
        )

if __name__ == "__main__":
    main()