import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def binary_cross_entropy(y, a):
    return -(y*np.log(a)+(1-y)*np.log(1-a))

def backward(x, y, a1, a2, W2):
    dz2 = a2 - y
    dW2 = np.outer(a1, dz2)
    db2 = dz2
    da1 = np.dot(W2, dz2)
    dz1 = da1 * a1*(1-a1)
    dW1 = np.outer(x, dz1)
    db1 = dz1
    return dW2, db2, dW1, db1



def update(W, b, dW, db, learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*db
    return W, b

x = np.array([1.5, 2.0])
y = 1
W1 = np.random.randn(2, 3) * 0.01
b1 = np.zeros(3)

W2 = np.random.randn(3, 1) * 0.01
b2 = np.zeros(1)


learning_rate = 0.1

for i in range(1000):
    z1 = np.dot(x, W1) +b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) +b2
    a2 = sigmoid(z2)

    loss = binary_cross_entropy(y, a2)

    dW2, db2, dW1, db1 = backward(x, y, a1, a2, W2)
    W1, b1 = update(W1, b1, dW1, db1, learning_rate)
    W2, b2 = update(W2, b2, dW2, db2, learning_rate)

    if i % 100 == 0:
        print(loss)


