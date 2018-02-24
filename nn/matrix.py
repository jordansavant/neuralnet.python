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

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.values[j][i] = self.values[i][j]
        return result

    def randomize(self):
        self.map(lambda x: random.randrange(0, 10))
        return

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
        self.elementwise_map(lambda i, j, x: m.values[i][j] + x)
        return

    def elementwise_subtract(self, m):
        return

    def elementwise_multiply(self, m):
        return

    def elementwise_divide(self, m):
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
        return

    @staticmethod
    def from_list(l):
        return

    @staticmethod
    def to_list(m):
        return

    @staticmethod
    def clone(m):
        return