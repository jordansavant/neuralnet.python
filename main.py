from nn.matrix import Matrix
from nn.neuralnet import NeuralNetwork

# training XOR
training_data = [
    {
        'inputs': [1.0, 0.0],
        'targets': [1.0]
    },
    {
        'inputs': [0.0, 1.0],
        'targets': [1.0]
    },
    {
        'inputs': [1.0, 1.0],
        'targets': [0.0]
    },
    {
        'inputs': [0.0, 0.0],
        'targets': [0.0]
    }
]
nn = NeuralNetwork(2, 2, 1, 0.1)

# Train it
for i in range(10000):
    for d in training_data:
        input_matrix = Matrix(2, 1)
        input_matrix[0][0] = d['inputs'][0]
        input_matrix[1][0] = d['inputs'][1]
        target_matrix = Matrix(1, 1)
        target_matrix[0][0] = d['targets'][0]

        nn.train(input_matrix, target_matrix)

# Test it
input_matrix = Matrix(2, 1)
input_matrix[0][0] = 1.0
input_matrix[1][0] = 0.0
print(input_matrix, nn.feedforward(input_matrix))

input_matrix = Matrix(2, 1)
input_matrix[0][0] = 0.0
input_matrix[1][0] = 1.0
print(input_matrix, nn.feedforward(input_matrix))

input_matrix = Matrix(2, 1)
input_matrix[0][0] = 1.0
input_matrix[1][0] = 1.0
print(input_matrix, nn.feedforward(input_matrix))

input_matrix = Matrix(2, 1)
input_matrix[0][0] = 0.0
input_matrix[1][0] = 0.0
print(input_matrix, nn.feedforward(input_matrix))

# nn = NeuralNetwork(2, 2, 1, 0.01)
# print(nn)
# input_matrix = Matrix(2, 1)
# input_matrix[0][0] = 1
# input_matrix[1][0] = 2
# print(input_matrix)
# output_matrix = nn.feedforward(input_matrix)
# print(output_matrix)

# m = Matrix(2, 3)
# print("m", m)
# mt = m.transpose()
# print("mt", mt)
# mt.randomize()
# print("mt.randomize()", mt)
# m.scalar_add(5)
# print("m.scalar_add(5)", m)
# m.scalar_subtract(4)
# print("m.scalar_subtract(4)", m)
# m.scalar_divide(2)
# print("m.scalar_divide(2)", m)
# m.scalar_multiply(3)
# print("m.scalar_multiply(3)", m)
# m.map(lambda x: int(x))
# print("m.map(lambda x: int(x))", m)

# m2 = Matrix(2, 3)
# print("m2", m2)
# m2.elementwise_add(m)
# print("m2.elementwise_add(m)", m2)

# m3 = Matrix.clone(m2)
# print("m3 = Matrix.clone(m2)", m3)
# m3.randomize()
# print("m3.randomize()", m3)
# print("m2", m2)

# print("m2", m3)
# print("m3[1][2]", m3[1][2])
# print("m3[0][0]", m3[0][0])

# m1 = Matrix(2,3)
# m1[1][1] = 2
# m2 = Matrix(3,2)
# m2[0][1] = 3
# m3 = Matrix.dot_product(m1, m2)
# print(m1, m2, m3)

# m1.randomize_range(-1, 1)
# print(m1)