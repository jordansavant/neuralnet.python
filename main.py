from nn.matrix import Matrix

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

m1 = Matrix(2,3)
m1[1][1] = 2
m2 = Matrix(3,2)
m2[0][1] = 3
m3 = Matrix.dot_product(m1, m2)
print(m1, m2, m3)

