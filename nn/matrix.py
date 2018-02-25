import random

class Matrix(object):

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.values = []
        for i in range(self.rows):
            self.values.append([])
            for j in range(self.cols):
                self.values[i].append(1)

    def __str__(self):
        s = ""
        for c in self.values:
            s += "| "
            for n in c:
                s += str(n) + " "
            s = s.strip() +" |\n"
        return "rows {}, cols {}\n{}".format(self.rows, self.cols, s)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.values[j][i] = self.values[i][j]
        return result

    def randomize(self):
        self.map(lambda x: random.randrange(0, 1))
        return
    
    def randomize_range(self, min, max):
        self.map(lambda x: random.uniform(min, max))

    def scalar_add(self, n):
        self.map(lambda x: x + n)
        return
    
    def scalar_subtract(self, n):
        self.map(lambda x: x - n)
        return

    def scalar_multiply(self, n):
        self.map(lambda x: x * n)
        return

    def scalar_divide(self, n):
        self.map(lambda x: x / n)
        return

    def elementwise_add(self, m):
        self.elementwise_map(lambda i, j, x: x + m.values[i][j])
        return

    def elementwise_subtract(self, m):
        self.elementwise_map(lambda i, j, x: x - m.values[i][j])
        return

    def elementwise_multiply(self, m):
        self.elementwise_map(lambda i, j, x: x * m.values[i][j])
        return

    def elementwise_divide(self, m):
        self.elementwise_map(lambda i, j, x: x / m.values[i][j])
        return
    
    def map(self, func):
        for i, c in enumerate(self.values):
            for j, n in enumerate(self.values[i]):
                self.values[i][j] = func(n)
        return
    
    def elementwise_map(self, func):
        for i, c in enumerate(self.values):
            for j, n in enumerate(self.values[i]):
                self.values[i][j] = func(i, j, n)
        return

    @staticmethod
    def dot_product(a, b):
        if a.cols != b.rows:
            raise Exception("matrix a columns != matrix b rows")
        
        result = Matrix(a.rows, b.cols)
        for i, c in enumerate(result.values):
            for j, d in enumerate(result.values[i]):
                # dot product of values in column
                sum = 0
                for k, e in enumerate(a.values[i]):
                    sum += a.values[i][k] * b.values[k][j]
                result.values[i][j] = sum
        return result

    @staticmethod
    def from_list(l):
        result = Matrix(len(l), 1)
        for i, v in enumerate(l):
            result.values[i][0] = l[i]
        return result

    @staticmethod
    def to_list(m):
        result = []
        for i, c in enumerate(m.values):
            for j, n in enumerate(m.values[i]):
                result.append(m.values[i][j])
        return result

    @staticmethod
    def clone(m):
        result = Matrix(m.rows, m.cols)
        result.elementwise_map(lambda i, j, x: m.values[i][j])
        return result