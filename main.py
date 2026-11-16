import numpy as np

TRAIN = [
    [0, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0],
]

D = 0.001
LR = 0.1

INS = 2
MIDS = 2
OUTS = 1

def loss():
    total_cost = 0
    for tr in TRAIN:
        input[0] = tr[:2]
        calc()
        total_cost += cost(tr[2], l2[0])
    return total_cost

def cost(a, b):
    return (a-b)**2

def calc():
    global l1, l2
    l1 = layer(input, w1, b1)
    l2 = layer(l1, w2, b2)

def layer(a, w, b):
    return sig(a @ w + b)

def sig(x):
    return 1 / (1 + np.exp(-x))

w1 = np.random.rand(INS, MIDS)
b1 = np.random.rand(1, MIDS)
l1 = np.zeros_like(b1)

w2 = np.random.rand(MIDS, OUTS)
b2 = np.random.rand(1, OUTS)
l2 = np.zeros_like(b2)

input = np.zeros((1, INS))
target = np.zeros_like(l2)

for i in range(1000*1000):
    if i%1000==0: print(i, loss())
    for tr_idx, tr in enumerate(TRAIN):
        input[0] = tr[:2]
        calc()
        target[0] = tr[2]

    # ∂cost/∂a2 = 2(a2 - target)
        _da2 = 2*(l2 - target)

    # ∂cost/∂z2 = ∂cost/∂a2 ⊙ (a2 ⊙ (1 - a2))
        _dz2 = _da2 * (l2 * (1 - l2))
    # ∂cost/∂w2 = a1ᵀ · ∂cost/∂z2
        _dw2 = l1.T @ _dz2
    # ∂cost/∂b2 = ∂cost/∂z2
        _db2 = _dz2

    # ∂cost/∂a1 = ∂cost/∂z2 · w2ᵀ
        _da1 = _dz2 @ w2.T

    # ∂cost/∂z1 = ∂cost/∂a1 ⊙ (a1 ⊙ (1 - a1))
        _dz1 = _da1 * (l1 * (1 - l1))
    # ∂cost/∂w1 = aᵀ · ∂cost/∂z1
        _dw1 = input.T @ _dz1
    # ∂cost/∂b1 = ∂cost/∂z1
        _db1 = _dz1

        w2 -= _dw2 * LR
        b2 -= _db2 * LR
        w1 -= _dw1 * LR
        b1 -= _db1 * LR