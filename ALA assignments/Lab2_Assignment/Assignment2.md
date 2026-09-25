# ALA Assignment 2

## Semantic Tagging with GloVe and NumPy

### Token Statistics

- Tokens before preprocessing: 81
- Tokens after preprocessing: 64
- In-vocabulary tokens: 58
- OOV tokens: 6

### Matrix Shapes

- T.shape = (20, 50)
- W.shape = (58, 50)
- S_vector.shape = (58, 20)
- S_numpy.shape = (58, 20)

### Verification

- np.allclose = True
- Maximum absolute difference = 5.5443353796924555e-08

## TOP 8 TAGS from my example 
----------
Rank  Tag                   Score       Best-matching word
-----------------------------------------------------------------
1     research              1.0000      research
2     innovation            1.0000      innovation
3     students              1.0000      students
4     technology            1.0000      technology
5     faculty               1.0000      faculty
6     collaboration         1.0000      collaboration
7     university            0.8258      faculty
8     entrepreneurship      0.7957      innovation