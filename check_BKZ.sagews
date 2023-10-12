︠97677adf-120b-41de-8edf-91d0c3776b54︠
import numpy as np
from sage.modules.free_module_integer import IntegerLattice

def Gram_Schmidt(b):
  '''
  入力　b: n次元格子の基（行列）
  出力　GSOb: 入力基のGSOベクトル（行列）
        mu: 入力基のGSO係数（行列）
  '''
  m = b.ncols()
  n = b.nrows()
  GSOb = zero_matrix(QQ, n, m)
  mu = identity_matrix(QQ, n)
  for i in range(n):
    GSOb[i] = b[i][:]
    for j in range(i):
      mu[i, j] = b[i].inner_product(GSOb[j]) / (GSOb[j].norm(2) ^ 2)
      GSOb[i] -= mu[i, j] * GSOb[j][:]
  return GSOb, mu

#二乗normの作成
def make_square_norm(GSOb):
  '''
  入力　GSOb:n次元格子の基（行列）
  出力　B:GSObの各横vectorの二乗normを並べたもの（vector）
  '''
  n = GSOb.nrows()
  B = zero_vector(QQ, n)
  for i in range(n):
    B[i] = GSOb[i].norm(2) ^ 2
  return B

#与えられた上界より小さいvectorの数え上げ
def ENUM(mu, B, R):
  '''
  入力　mu：GSO-vector（行列）
  　　　B：GSO-vectorの二乗norm
  　　　R：数え上げ上界列
  出力　短いvectorの係数vector
  '''
  n = len(B)
  sigma = zero_matrix(QQ, n + 1, n)
  r = np.arange(n + 1); r -= 1; r = np.roll(r, -1)
  rho = zero_vector(QQ, n + 1)
  v = zero_vector(ZZ, n); v[0] = 1
  c = zero_vector(QQ, n)
  w = zero_vector(ZZ, n)
  last_nonzero = 0
  k = 0
  while 1:
    rho[k] = rho[k + 1] + (v[k] - c[k]) ^ 2 * B[k]
    if rho[k] <= R ** 2:
      if k == 0:
        return v
      k -= 1
      r[k - 1] = max(r[k - 1], r[k])
      for i in range(k + 1, r[k] + 1)[::-1]:
        sigma[i, k] = sigma[i + 1, k] + mu[i, k] * v[i]
      c[k] = -sigma[k + 1, k]
      v[k] = round(c[k])
      w[k] = 1
    else:
      k += 1
      if k == n:
        u = zero_vector(ZZ, n)
        return u
      r[k - 1] = k
      if k >= last_nonzero:
        last_nonzero = k
        v[k] += 1
      else:
        if v[k] > c[k]:
          v[k] -= w[k]
        else:
          v[k] += w[k]
        w[k] += 1
    #print("********\nv =",v,"\nvector =",coef2lattice(v, b),"\n||vector|| =",np.linalg.norm(coef2lattice(v, b)),"\n********\n")

#最短vectorの数え上げ
def ENUM_all(mu, B, R, eps):
  n = mu.nrows()
  ENUM_v = zero_vector(ZZ, n)
  pre_ENUM_v = zero_vector(ZZ, n)
  x = zero_matrix(ZZ, n)
  while 1:
    pre_ENUM_v = ENUM_v[:]
    ENUM_v = ENUM(mu, B, R)
    if np.all(ENUM_v == 0):
      return pre_ENUM_v
    R *= eps
    #print("R=", R)

#A function that outputs basis over projected lattice $\pi_k(L)$
def project_basis(k, l, b):
  """射影格子の基{π_k(b_k),…,π_k(b_l)}を出力します。"""
  n = b.nrows()
  m = b.ncols()
  GSOb, mu = Gram_Schmidt(b)
  pi_b = zero_matrix(QQ, l - k + 1, m)
  for i in range(k, l + 1):
    for j in range(k, n):
      pi_b[i - k] += (b[i].inner_product(GSOb[j])) / (GSOb[j].norm(2)) ^ 2 * GSOb[j]
  return pi_b

#A function that checks whether input basis are HKZ-reduced or not
def check_HKZ(b):
  n = b.nrows()
  m = b.ncols()
  GSOb, mu = Gram_Schmidt(b)
  for i in range(n):
    proj_b = project_basis(i, n - 1, b) #射影格子の基vector
    proj_GSOb, mu = Gram_Schmidt(proj_b) #射影格子の基vectorのGSO情報
    B = make_square_norm(proj_GSOb) #射影格子の基vectorのGSO-vectorの二乗norm
    R = proj_b[0].norm(2) #数え上げ上界
    coef_vec = ENUM_all(mu, B, R, 0.99)
    lat_vec = zero_vector(QQ, m)
    ###射影格子上の最短非零vector###
    for j in range(n - i):
      lat_vec += coef_vec[j] * proj_b[j][:]
    ######
    if GSOb[i].norm(2) != lat_vec.norm(2):
      return False
  return True

#A function that checks whether input basis are BKZ-reduced or not
def check_BKZ(b, beta):
  n = b.nrows()
  m = b.ncols()
  for i in range(n - beta + 1):
    proj_b = project_basis(i, beta + i - 1, b)
    if not check_HKZ(proj_b):
      return False
  return True

n = 20
b = identity_matrix(ZZ, n)
for i in range(n):
  b[i, 0] = randint(-99, 99)

b = b.BKZ(beta = 10)
print(b)
print("whether BKZ-reduced or not:",check_BKZ(b, 10))
︡dce893ac-1bc4-4bcc-8927-a7db94c96c5b︡{"stdout":"[ 0  0  0  0  0 -1  0  0  0  0  0  1  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0  0 -1  1  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0 -1  0  0  0  1  0  0  0  1  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  1]\n[ 0  0  0 -1  0  0  0  0 -1  1  0  0  0  0  0  0  0  0  0  0]\n[-1  0  0  0  0 -1  0  0  1  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  1  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0 -1  0  0  1  0  1  0  0  0  0  0  0  1  0  0]\n[ 1  0  0  0 -1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]\n[-1  0  0  0  0  0  0  0 -1  0  0  1  0  0  1  0  0  0  0  0]\n[ 0  0  0  0  0  1  1  0  1  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0]\n[ 0  0  0  0  1  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  1  0 -1  1  0  0  1  0  0  0  0  0  0  0]\n[-1 -1  0  0 -1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0 -1 -1  0  0  1  0  0  0  0  1  0  0]\n[ 0  0  0  0  0  0  0  0 -1  0  0 -1  0  0  0  1  0  1  0  0]\n[ 0  0 -1  0  0  0  0  0 -1  0  0  1  0  0 -1  0  0  0  0  0]\n[ 0  0  0  0  0 -1  0  0 -1 -1  0  0  0  0  0  0  0  0  1  0]\n[ 0  0  0  0  0  0 -1 -1  1  0  1  0  0  0  0  0  0  0  0  0]\n"}︡{"stdout":"whether size-reduced or not: True\n"}︡{"stdout":"whether BKZ-reduced or not:"}︡{"stdout":" True\n"}︡{"done":true}
︠36edcb19-68a3-4675-9749-afbbed95759d︠
︡e8b4dfb7-d1c6-466b-a934-d5c6ac571be5︡{"done":true}
︠6b791023-e0b3-4a37-9c78-1b2bb7e14c48︠
︡c744d3e0-d2b9-49bb-99cb-a416bd62511c︡{"done":true}
︠6c4c378e-ac86-4f6d-866b-0e4f929ba25e︠









