import string
import numpy as np
import gensim.downloader as api
from vec import Vec

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


# STOPWORDS


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from", "it", "this",
    "that", "these", "those", "our", "we", "they",
    "their", "them", "as", "be", "been", "being",
    "has", "have", "had", "will", "would", "can",
    "could", "about", "into", "through", "its"
}



# PART C - TEXT PREPROCESSING


def preprocess_text(text, model):
    """
    Preprocess the input text.

    Steps:
    1. lowercase
    2. remove punctuation
    3. split into tokens
    4. remove stopwords
    5. remove tokens of length <= 2
    6. separate in-vocabulary and OOV tokens
    """

    # Number of tokens before preprocessing
    text_lower = text.lower()

    # Remove punctuation
    text_no_punctuation = text_lower.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split into tokens
    tokens = text_no_punctuation.split()

    tokens_before = len(tokens)

    # Remove stopwords
    filtered_tokens = [
        token for token in tokens
        if token not in STOPWORDS
    ]

    # Remove tokens whose length <= 2
    filtered_tokens = [
        token for token in filtered_tokens
        if len(token) > 2
    ]

    # In-vocabulary and OOV
    in_vocab_tokens = []
    oov_tokens = []

    for token in filtered_tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)

    statistics = {
        "before_preprocessing": tokens_before,
        "after_preprocessing": len(filtered_tokens),
        "in_vocab": len(in_vocab_tokens),
        "out_of_vocab": len(oov_tokens),
        "distinct_oov": sorted(set(oov_tokens))
    }

    return filtered_tokens, in_vocab_tokens, oov_tokens, statistics



# PART D - TAG MATRIX


def build_tag_matrix(model, tags):
    """
    Construct T with shape (20, 50).

    Single-word tag:
        vector = model[tag]

    Multi-word tag:
        arithmetic mean of component word vectors.
    """

    if len(tags) != 20:
        raise ValueError("Exactly 20 tags are required.")

    tag_names = list(tags)
    rows = []

    for tag in tags:

        words = tag.lower().split()

        vectors = []

        for word in words:
            if word not in model.key_to_index:
                raise ValueError(
                    f"Tag word not in vocabulary: {word}"
                )

            vectors.append(model[word])

        vector = np.mean(vectors, axis=0)
        rows.append(vector)

    T = np.array(rows, dtype=float)

    if T.shape != (20, 50):
        raise ValueError(
            f"Unexpected tag matrix shape: {T.shape}"
        )

    return tag_names, T



# PART D - TEXT MATRIX


def build_text_matrix(model, tokens):
    """
    Construct W.

    Returns:
        in_vocab_tokens
        oov_tokens
        W
    """

    in_vocab_tokens = []
    oov_tokens = []
    rows = []

    for token in tokens:

        if token in model.key_to_index:
            in_vocab_tokens.append(token)
            rows.append(model[token])
        else:
            oov_tokens.append(token)

    if len(rows) == 0:
        raise ValueError(
            "No in-vocabulary words remain after preprocessing."
        )

    W = np.array(rows, dtype=float)

    if W.shape[1] != 50:
        raise ValueError(
            f"Expected 50-dimensional vectors, got {W.shape}"
        )

    return in_vocab_tokens, oov_tokens, W



# PART E - VECTOR CLASS SIMILARITY


def similarity_matrix_vector_class(W, T):
    """
    Compute cosine similarity using the Vec class.

    No NumPy matrix multiplication is used here.
    """

    n_words = W.shape[0]
    n_tags = T.shape[0]

    S = []

    for i in range(n_words):

        row = []

        for j in range(n_tags):

            word_vector = Vec(W[i].tolist())
            tag_vector = Vec(T[j].tolist())

            similarity = word_vector.cosine_similarity(tag_vector)

            row.append(similarity)

        S.append(row)

    return np.array(S, dtype=float)



# PART F - NUMPY SIMILARITY


def similarity_matrix_numpy(W, T):
    """
    Compute cosine similarities using NumPy.

    No Python-level double loop is used.
    """

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

    if np.any(W_norm == 0):
        raise ValueError("W contains a zero vector.")

    if np.any(T_norm == 0):
        raise ValueError("T contains a zero vector.")

    W_hat = W / W_norm
    T_hat = T / T_norm

    S = W_hat @ T_hat.T

    return S



# PART H - MAX POOLING


def rank_tags(tag_names, S, in_vocab_tokens):
    """
    Rank tags using max pooling.

    For each tag:

        score[j] = max(S[:, j])

    Also return the text word that produced
    the maximum similarity.
    """

    rankings = []

    for j, tag in enumerate(tag_names):

        column = S[:, j]

        best_index = int(np.argmax(column))
        best_score = float(column[best_index])
        best_word = in_vocab_tokens[best_index]

        rankings.append(
            (tag, best_score, best_word)
        )

    rankings.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return rankings








def main():

    print("Loading GloVe model...")

    model = api.load("glove-wiki-gigaword-50")

    print("Model loaded successfully.")
    print()

    # Read text file

    with open(
        "manipal_text.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Preprocessing
  

    processed_tokens, in_vocab_tokens, oov_tokens, stats = \
        preprocess_text(text, model)

    print("TOKEN STATISTICS")
    print("----------------")
    print(
        "Tokens before preprocessing:",
        stats["before_preprocessing"]
    )

    print(
        "Tokens after preprocessing:",
        stats["after_preprocessing"]
    )

    print(
        "In-vocabulary tokens:",
        stats["in_vocab"]
    )

    print(
        "Out-of-vocabulary tokens:",
        stats["out_of_vocab"]
    )

    print(
        "Distinct OOV tokens:",
        stats["distinct_oov"]
    )

    print()

    # Build T
   

    tag_names, T = build_tag_matrix(
        model,
        TAGS
    )

    # Build W


    in_vocab_tokens, oov_tokens, W = \
        build_text_matrix(
            model,
            processed_tokens
        )

    print("MATRIX SHAPES")
    print("-------------")
    print("T.shape =", T.shape)
    print("W.shape =", W.shape)
    print()

  
    # Vector class similarity


    S_vector = similarity_matrix_vector_class(
        W,
        T
    )

    print(
        "S_vector.shape =",
        S_vector.shape
    )

    # NumPy similarity
  

    S_numpy = similarity_matrix_numpy(
        W,
        T
    )

    print(
        "S_numpy.shape =",
        S_numpy.shape
    )

 
    # Verification
   

    same = np.allclose(
        S_vector,
        S_numpy,
        atol=1e-6
    )

    max_difference = np.max(
        np.abs(
            S_vector - S_numpy
        )
    )

    print()
    print("VERIFICATION")
    print("------------")
    print("np.allclose =", same)
    print(
        "Maximum absolute difference =",
        max_difference
    )

    
    # Max-pool ranking

    rankings = rank_tags(
        tag_names,
        S_numpy,
        in_vocab_tokens
    )

    print()
    print("TOP 8 TAGS")
    print("----------")

    print(
        f"{'Rank':<6}"
        f"{'Tag':<22}"
        f"{'Score':<12}"
        f"{'Best-matching word'}"
    )

    print("-" * 65)

    for rank, (tag, score, word) in enumerate(
        rankings[:8],
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{tag:<22}"
            f"{score:<12.4f}"
            f"{word}"
        )


if __name__ == "__main__":
    main()