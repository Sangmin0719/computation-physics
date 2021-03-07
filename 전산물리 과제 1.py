import numpy as np
import matplotlib.pyplot as plt

# github 주소 입니다. https://github.com/Sangmin0719/computation-physics

#3.
ey = np.array([0,1]) #단위 백터를 nparray로 받는다.
for a in np.arange(-2,2,0.2): #변수 a가 바뀔때 마다 그림을 그려주기위해 for in np.range를 사용했습니다. 
    A = np.array([[0,a],      # A와 ey를 연산하면, (a,1)이 나오는데 이 값은 a값에 상관 없이 y = 1위에 있습니다.
                 [0,1]])
    v = np.dot(A,ey)          # v = (a,1)이 나옵니다.
    plt.arrow(0,0,v[0],v[1], head_width = 0.02, color = 'r') # 해당 백터를 arraw를 이용하여 그려줍니다.

def line(k):                  #그린 백터들의 종점이 y = 1에 있는지 보기위해, 그림에 y = 1도 같이 그립니다.
    return 0*k+1              # return 값이 1인 함수를 생성합니다.  
x = np.linspace(-2,2,10)      # x를 -2부터 2까지 10등분한 값들을 받습니다.

plt.plot(x,line(x))           # line(x)를 그립니다. 즉 y = 1을 그려줍니다.

plt.show()

#4.
# u+v = 2ex, u-v = 2ey, 
# Au+Av = A(u+v) = 2Aex = (2,0),  Aex = (1,0)  -ㄱ
# Au-Av = A(u-v) = 2Aey = (4,4),  Aey = (2,2)  -ㄴ
# A를 ex에 연산 한 결과가 백터로 표현 가능하므로 A는 2*2행렬이다.
# A = array([[a,b],   로 두고, ㄱ,ㄴ의 식을 써주면 다음과 같다.
#            [c,d]])
# 1*a + 0*b + 0*c + 0*d = 1
# 0*a + 0*b + 1*c + 0*d = 0
# 0*a + 1*b + 0*c + 0*d = 2
# 0*a + 0*b + 0*c + 1*d = 2
# 위식을 Bx = b로 표현 할수 있고 그때의 B와 b는 다음과 같다. x = [a,b,c,d]이다.
B = np.array([[1,0,0,0],
              [0,0,1,0],
              [0,1,0,0],
              [0,0,0,1]])

b = np.array([1,0,2,2])

sol = np.linalg.solve(B,b) # linalg.solve(A,b)는 Bx = b를 만족하는 x를 sol로 받습니다.
print('sol:',sol) #sol을 출력해줍니다. sol:[1,2,0,4]가 출력됩니다.
A = np.array([[1,2],
              [0,2]]) # A는 다음과 같이 구할 수 있습니다. 

#5. 토끼의 수를 x1, 닭의 수를 x2 라고 두겠습니다.
# x1+x2 = 40, 4x1 +2x2 = 92 이식을 만족해야합니다.
# 위 식을 행렬로 표현하면 Ax = b 로 표현할 수 있고, A와 b는 다음과 같습니다.

A = np.array([[1,1],
              [4,2]]) 
b = np.array([40,92])

sol = np.linalg.solve(A,b) # linalg.solve(A,b)는 Ax = b를 만족하는 x를 sol로 받습니다.
print('sol:',sol) #sol을 출력해줍니다.
#출력되는 sol은 [6,34] 입니다.
