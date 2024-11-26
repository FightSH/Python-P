import torch
# 标量由只有一个元素的张量表示
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x ** y)


# 向量可以被视为标量值组成的列表 一维张量表示向量
x = torch.arange(4)
print(x)

# 向量的长度维度和形状 向量只是一个数字数组
# 向量的长度通常称为向量的维度
print(len(x))

# 当用张量表示一个向量（只有一个轴）时，我们也可以通过.shape属性访问向量的长度。
# 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于(只有一个轴的张量，形状只有一个元素。)

# 向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。 然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。
print(x.shape)


# 矩阵，两个轴的张量
A = torch.arange(20).reshape(5, 4)
print(A)




# 向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构
# 张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的n维数组的通用方法
X = torch.arange(24).reshape(2, 3, 4)
X


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
# 两个矩阵的按元素乘法称为Hadamard积
A * B

# 点积
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

# 矩阵乘法

