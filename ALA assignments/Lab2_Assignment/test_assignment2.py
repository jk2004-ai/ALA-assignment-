import numpy as np
import pytest

from assignment2 import (
    TAGS,
    build_tag_matrix,
    build_text_matrix,
    similarity_matrix_vector_class,
    similarity_matrix_numpy,
    rank_tags
)



# Fake model for testing


class FakeModel:

    def __init__(self):
        self.key_to_index = {
            word: i
            for i, word in enumerate(
                TAGS + [
                    "hello",
                    "world",
                    "computer",
                    "science",
                    "student",
                    "learning"
                ]
            )
        }

        self.vectors = {}

        for i, word in enumerate(self.key_to_index):
            vector = np.zeros(50, dtype=float)

            # Give every word a non-zero vector
            vector[i % 50] = 1.0

            self.vectors[word] = vector

    def __getitem__(self, word):
        return self.vectors[word]


@pytest.fixture
def model():
    return FakeModel()



# TEST 1
# T must have shape (20, 50)


def test_tag_matrix_shape(model):

    tag_names, T = build_tag_matrix(
        model,
        TAGS
    )

    assert len(tag_names) == 20
    assert T.shape == (20, 50)



# TEST 2
# W must have shape (n, 50)


def test_text_matrix_shape(model):

    tokens = [
        "hello",
        "world",
        "student"
    ]

    in_vocab, oov, W = build_text_matrix(
        model,
        tokens
    )

    assert len(in_vocab) == 3
    assert W.shape == (3, 50)



# TEST 3
# OOV token excluded


def test_oov_token(model):

    tokens = [
        "hello",
        "unknownword",
        "student"
    ]

    in_vocab, oov, W = build_text_matrix(
        model,
        tokens
    )

    assert "unknownword" in oov
    assert "unknownword" not in in_vocab
    assert W.shape == (2, 50)


# TEST 4
# S shape


def test_similarity_matrix_shape(model):

    _, T = build_tag_matrix(
        model,
        TAGS
    )

    tokens = [
        "hello",
        "world",
        "student"
    ]

    _, _, W = build_text_matrix(
        model,
        tokens
    )

    S = similarity_matrix_numpy(
        W,
        T
    )

    assert S.shape == (3, 20)



# TEST 5
# Vector and NumPy implementations agree


def test_vector_and_numpy_similarity_agree(model):

    _, T = build_tag_matrix(
        model,
        TAGS
    )

    tokens = [
        "hello",
        "world",
        "student"
    ]

    in_vocab, _, W = build_text_matrix(
        model,
        tokens
    )

    S_vector = similarity_matrix_vector_class(
        W,
        T
    )

    S_numpy = similarity_matrix_numpy(
        W,
        T
    )

    assert np.allclose(
        S_vector,
        S_numpy,
        atol=1e-6
    )



# TEST 6
# Individual cosine similarity agrees


def test_individual_cosine_similarity(model):

    _, T = build_tag_matrix(
        model,
        TAGS
    )

    tokens = [
        "hello"
    ]

    in_vocab, _, W = build_text_matrix(
        model,
        tokens
    )

    S = similarity_matrix_numpy(
        W,
        T
    )

    from vec import Vec

    word_vector = Vec(
        W[0].tolist()
    )

    tag_vector = Vec(
        T[0].tolist()
    )

    expected = word_vector.cosine_similarity(
        tag_vector
    )

    assert S[0, 0] == pytest.approx(
        expected
    )


# TEST 7
# Invalid tag must raise error

def test_invalid_tag_raises_error(model):

    invalid_tags = TAGS.copy()

    invalid_tags[-1] = "this_tag_does_not_exist"

    with pytest.raises(ValueError):

        build_tag_matrix(
            model,
            invalid_tags
        )


# TEST 8
# Repeated words must be preserved


def test_repeated_words_preserved(model):

    tokens = [
        "student",
        "student",
        "hello"
    ]

    in_vocab, _, W = build_text_matrix(
        model,
        tokens
    )

    assert in_vocab == [
        "student",
        "student",
        "hello"
    ]

    assert W.shape == (3, 50)



# TEST 9
# Ranking returns all tags


def test_rank_tags(model):

    _, T = build_tag_matrix(
        model,
        TAGS
    )

    tokens = [
        "hello",
        "world",
        "student"
    ]

    in_vocab, _, W = build_text_matrix(
        model,
        tokens
    )

    S = similarity_matrix_numpy(
        W,
        T
    )

    rankings = rank_tags(
        TAGS,
        S,
        in_vocab
    )

    assert len(rankings) == 20

    for tag, score, word in rankings:

        assert tag in TAGS
        assert isinstance(score, float)
        assert word in in_vocab