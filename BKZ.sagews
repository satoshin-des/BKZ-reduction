︠8168814e-6055-4927-abf3-516d473fb61cs︠
import numpy as np
import copy
from time import perf_counter

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

def coef2lattice(v, b):
  n = b.nrows()
  m = b.ncols()
  x = zero_vector(ZZ, m)
  for i in range(n):
    x += v[i] * b[i][:]
  return x

#体積
def vol(B):
  n = B.nrows()
  C = zero_matrix(ZZ, n, n)
  C = B * B.transpose()
  return sqrt(C.det())

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
  n = b.nrows()
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

#射影格子の基底の出力
def project_basis(k, l, b):
  n = b.nrows()
  m = b.ncols()
  GSOb, mu = Gram_Schmidt(b)
  pi_b = zero_matrix(QQ, l - k + 1, m)
  for i in range(k, l + 1):
    for j in range(k, n):
      pi_b[i - k] += (b[i].inner_product(GSOb[j])) / (GSOb[j].norm(2)) ^ 2 * GSOb[j]
  return pi_b

def norm_pi(k, l, v, mu, GSOb):
  m = GSOb.ncols()
  pi_b = zero_vector(QQ, m)
  for i in range(k - 1, l):
    for j in range(i + 1):
      pi_b += v[i - k + 1] * mu[i, j] * GSOb[j][:]
  return pi_b.norm(2)

#BKZ
def BKZ(b, beta, d):
  n = b.nrows()
  m = b.ncols()
  b = b.LLL(delta = d)
  z = 0; k = 0
  eps = 0.99
  while(z < n - 1):
    #print(z)
    k = k % (n - 1) + 1
    l = min(k + beta - 1, n)
    h = min(l + 1, n)
    w = zero_vector(ZZ, l - k + 1)
    v = zero_vector(ZZ, m)
    s = zero_vector(QQ, m)
    GSOb, mu = Gram_Schmidt(b)
    B = make_square_norm(GSOb)
    proj_b = project_basis(k - 1, l - 1, b) #射影格子の基
    proj_GSOb, proj_mu = Gram_Schmidt(proj_b)
    proj_B = make_square_norm(proj_GSOb)
    R = proj_b[0].norm(2)
    w = ENUM_all(proj_mu, proj_B, R, eps)#部分射影格子L_{[k,l]}上の最短vectorの係数vector
    for i in range(l - k + 1):
      v += w[i] * b[i + k - 1][:]
      s += w[i] * proj_b[i][:]
    if (B[k - 1] > s.norm(2) ^ 2):
      z = 0
      #***k番目のvectorとしてvを挿入し、MLLLで一次従属性を取り除く***
      c = zero_matrix(ZZ, h + 1, m)
      for i in range(k - 1):
        c[i] = b[i][:]
      c[k - 1] = v[:]
      for i in range(k, h + 1):
        c[i] = b[i - 1][:]
      c = c.LLL(delta = d)
      for i in range(1, h + 1):
        b[i - 1] = c[i][:]
      #******
    else:
      z += 1
      #c = zero_matrix(ZZ, h, m)
      #for i in range(h):
      #  c[i] = b[i][:]
      #c = c.LLL(delta = d)
      #for i in range(h):
      #  b[i] = c[i][:]
      b = b.LLL(delta = d)
  return b

#HKZ簡約基か否かの判定
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
      print(i, GSOb[i], lat_vec)
      return False
  return True

#BKZ簡約基か否かの判定
def check_BKZ(b, beta):
  n = b.nrows()
  m = b.ncols()
  for i in range(n - beta + 1):
    proj_b = project_basis(i, beta + i - 1, b)
    if not check_HKZ(proj_b):
      return False
  return True

#main
random = True
n = 20 #格子次元
beta = 10 #blocksize
delta = 0.99 #簡約parametre
b = zero_matrix(ZZ, n, n)

if random:
  b = identity_matrix(ZZ, n)
  for i in range(n):
    b[i, 0] = randint(-99, 99)
else:#自分で設定してください
  #SVP challenge
  pass

print("入力基行列：\n",b)

start_time = perf_counter()
BKZb = BKZ(b, beta, delta)
end_time = perf_counter()

print("\n",beta,"-BKZ基行列：\n",BKZb)
print("Run time is ",end_time - start_time,"[secs]")
print(check_BKZ(BKZb, beta))
︡bed24bb5-a3bb-4519-9f9f-fd02ba1cf943︡{"stdout":"入力基行列：\n [  8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[-40   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[ 23   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[ 71   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[-49   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[ -6   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[ 78   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0]\n[-32   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0]\n[ 49   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0]\n[-45   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0]\n[ 16   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0]\n[ 20   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0]\n[-91   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0]\n[-80   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0]\n[ 40   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0]\n[-85   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0]\n[ 26   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0]\n[-40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0]\n[ 18   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0]\n[-89   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1]\n"}︡{"stdout":"\n 10 -BKZ基行列：\n [ 0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0]\n[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0]\n[-1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[-1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[-1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[ 1  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]\n[ 1  0  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[-1  0  0 -1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0]\n[ 0  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0]\n[ 1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0]\n[ 1  0  0  0  0 -1  0  0  0  1  0  0  0  0  0  0  0  0  0  0]\n[ 0  0 -1  0  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0]\n[ 0  0  1  0  0 -1  0  0  0  0  0  0  0  0  0  1  0  0  0  0]\n[ 0  0  1  0  0 -1  0  0  0  0  0  0  1  0  0  0 -1  0  0  0]\n[ 1  0  0  0  1  0  1  0  0  0  0  1  0  0  0  0  0  0  0  0]\n"}︡{"stdout":"Run time is  6.629054540000652 [secs]\n"}︡{"stdout":"True"}︡{"stdout":"\n"}︡{"done":true}
︠923294c5-6bc6-421a-a433-e064c628396b︠



















