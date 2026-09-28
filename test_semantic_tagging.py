import numpy as np

from semantic_tagging import (
    TAGS,
    build_tag_matrix,
    build_text_matrix,
    similarity_matrix_numpy,
    similarity_matrix_vector_class,
    model
)

# ===========================
# Test Tag Matrix
# ===========================

tag_names, T = build_tag_matrix(
    model,
    TAGS
)

assert T.shape == (20, 50)

print("Test 1 Passed")

# ===========================
# Test Text Matrix
# ===========================

tokens = [
    "research",
    "innovation",
    "engineering",
    "xyzunknown"
]

in_vocab, oov, W = \
    build_text_matrix(
        model,
        tokens
    )

assert W.shape[1] == 50

print("Test 2 Passed")

# ===========================
# Test Similarity Shape
# ===========================

S_numpy = similarity_matrix_numpy(
    W,
    T
)

assert S_numpy.shape == (
    len(in_vocab),
    20
)

print("Test 3 Passed")

# ===========================
# Test Equality
# ===========================

S_vector = similarity_matrix_vector_class(
    W,
    T
)

assert np.allclose(
    S_vector,
    S_numpy
)

print("Test 4 Passed")

# ===========================
# Test OOV Detection
# ===========================

assert "xyzunknown" in oov

print("Test 5 Passed")

# ===========================
# Test Invalid Tag
# ===========================

try:

    bad_tags = TAGS[:-1]

    bad_tags.append(
        "abcdxyz"
    )

    build_tag_matrix(
        model,
        bad_tags
    )

    print(
        "Test Failed"
    )

except ValueError:

    print(
        "Test 6 Passed"
    )