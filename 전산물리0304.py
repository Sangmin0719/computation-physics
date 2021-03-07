#www.python.org
#www.sipy.org
#www.numpy.org 참고하세요!

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy
from numpy import cos, sin, exp #(굳이 np.안붙여도 sin을 불러올수 있음)
def func(x):
    return x+1 # 이런함수를  lambda로 표현가능

a = lambda x,y : x +y  #lambda 변수 : return값

print(sin(np.pi/2))

#단위 백터
ex = np.array([1,0])
ey = np.array([0,1])

plt.arrow(0,0,ex[0],ex[1],head_width = 0.2, color = 'b') #백터 그리기 arraw(시작x,시작y,끝x, 끝y color) 
plt.arrow(0,0,ey[0],ey[1],head_width = 0.2, color = 'r')
plt.xlim(-2,2) # x의 범위 지정
plt.ylim(-2,2)
plt.show()

# linear operator A(a*xvex +b*yvex)  = a*A*xvec + b*A*yvec 
# alpha가 -1,1까지 변할떄 
# A = ([1,alpha],[0,alpha])를 u = [1,1] 에 작용시키는 그림을 그려라
u= np.array([1,1])
for alpha in np.arange(-1,1,0.2):
    A = np.array([[1,alpha],
                 [0,alpha]])
    v = np.dot(A,u)
    plt.arrow(0,0,v[0],v[1],head_width = 0.2, color = 'r')

plt.xlim(-2,2) 
plt.ylim(-2,2)
plt.show()

# A = ([1,2],[0,3]) b = (5,4)일때 A*x = b를 성립하는 x를 찾으시오
# 역행렬이용, linalg.solve사용, 소거 법 사용

#역행렬 이용
A = np.array([[1,2],
             [0,3]])
b = np.array([5,4])

At = np.linalg.inv(A) # 역행렬을 구해주는 문
x1 = np.dot(At,b)
print(x1)

# linalg.solve이용
x2 = np.linalg.solve(A,b) 
print(x1)

# 연립방정식 풀이
# a1*x1 + a2*x
# x_A = xa0 + va0*t + a1*t^2/2
# x_b = xb0 + vb0*t + a2*t^2/2

# x_A = a1*t^2/2
# x_B = 100 +a2*t^2/2
# #같은방향
# a1*90^2/2 -a2*90^2/2 = 100
# #다른방향
# a1*30^2/2 +a2*30^2/2 = 100
t1 = 30
t2 = 90
A = np.array([[t1**2/2, t1**2/2],
              [t2**2/2, -t2**2/2]])   
b = np.array([100,100])

sol = np.linalg.solve(A,b)
print('sol', sol)

# A와B가 있는데 등속운동을 하고 있다. 서로의 초기 거리는 100M이다.
# 반대방향으로 걸을 때 걸리는 시간은 5초, 같은 방향으로 거리는 시간은 15초
# A,B의 속도를 찾으시오

t1 = 5
t2 = 15
A = np.array([[t1, t1],
              [t2, -t2]])
b = np.array([100,100])
sol = np.linalg.solve(A,b)
print('sol',sol)


#가우스 소거법
#Aij -> Aij - lambda * Ak, j = k, k+1,....n
# -lam*a11 + a21 =0 -> a21 - lam*a11 = 0
# lam =a21/a11

#(i,j) (행, 열)
#Loop in j 열을 스캔
#Loop in i 행을 스캔
# for j in range(0,n-1): # 소거하는 게 n-1번쨰 까지임, 예를들어 2by2는 1개만 소거하면 됨
#     for j in range(, )
#         lam = a[]

