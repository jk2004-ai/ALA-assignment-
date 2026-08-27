class Matrix:
    def __init__(self, data):
        if not data:
            raise ValueError("Matrix cannot be empty")

        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self):
        return f"Matrix({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    # Matrix addition
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")

        result = []

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)

        return Matrix(result)

    # Matrix subtraction
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions")

        result = []

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)

        return Matrix(result)

    # Scalar multiplication
    def __mul__(self, scalar):
        result = []

        for row in self.data:
            result.append([value * scalar for value in row])

        return Matrix(result)

    # Allows 3 * matrix
    def __rmul__(self, scalar):
        return self * scalar

    # Matrix-vector and matrix-matrix multiplication
    def __matmul__(self, other):
        # Matrix-vector multiplication
        if isinstance(other, list):
            if len(other) != self.cols:
                raise ValueError("Invalid vector size")

            return [
                sum(self.data[i][j] * other[j] for j in range(self.cols))
                for i in range(self.rows)
            ]

        # Matrix-matrix multiplication
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Invalid matrix dimensions")

            result = []

            for i in range(self.rows):
                row = []

                for j in range(other.cols):
                    value = sum(
                        self.data[i][k] * other.data[k][j]
                        for k in range(self.cols)
                    )

                    row.append(value)

                result.append(row)

            return Matrix(result)

        raise TypeError("Unsupported operand")


# Example usage
if __name__ == "__main__":
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    B = Matrix([
        [5, 6],
        [7, 8]
    ])

    print("A =", A)
    print("B =", B)

    print("A + B =", A + B)
    print("A - B =", A - B)
    print("A * 2 =", A * 2)

    vector = [5, 6]
    print("A @ vector =", A @ vector)

    print("A @ B =", A @ B)
