import numpy as np
import matplotlib.pyplot as plt
from sympy import *

p = lambda x, y: 2*x**2 + 2*y**2 + 4*x + 4*y + 3
q = lambda x, y: x**2 + y**2 + 2*x*y + 3*x + 5*y + 4

x = np.arange(-5, 5, 0.001)
y = np.arange(-5, 5, 0.001)

X, Y = np.meshgrid(x, y)

plt.contour(X, Y, p(X, Y), [0], colors='red')
plt.contour(X, Y, q(X, Y), [0], colors='blue')
plt.plot(-1.08405, 0, marker='x', color='black')
plt.plot(-0.29791, 0, marker='x', color='black')
plt.show()


# Using sylvester's method represent matrix Q_1
x = Symbol('x', real=True)
Q = Matrix([[2, 4, 2*x**2+4*x+3, 0], [0, 2, 4, 2*x**2+4*x+3], [1, 5+2*x, x**2+3*x+4, 0], [0, 1, 5+2*x, x**2+3*x+4]])
equation = Q.det()
print(equation)
roots_x = solve(equation, x)
print("Roots of {} are \n{} and {}".format(equation, roots_x[0].evalf(), roots_x[1].evalf()))


# Sanity check to see whether roots from ratio method are correct
# Ratio method roots were computed manually
y = Symbol('y', real= True)
P = 2*y**2 + 4*y + 2*x**2 + 4*x + 3
p_y1 = P.subs(x, roots_x[0])
p_y2 = P.subs(x, roots_x[1])
print("Polynomial y1 = {}".format(p_y1.evalf()))
print("Polynomial y2 = {}".format(p_y2.evalf()))
roots_y1 = solve(p_y1, y)
roots_y2 = solve(p_y2, y)
print("Roots y1 = {}, {}".format(roots_y1[0].evalf(), roots_y1[1].evalf()))
print("Roots y2 = {}, {}".format(roots_y2[0].evalf(), roots_y2[1].evalf()))


Q = x**2 + y**2 + 2*x*y + 3*x + 5*y + 4
q_y1 = Q.subs(x, roots_x[0])
q_y2 = Q.subs(x, roots_x[1])
print("Polynomial y3 = {}".format(q_y1.evalf()))
print("Polynomial y4 = {}".format(q_y2.evalf()))
roots_y3 = solve(q_y1, y)
roots_y4 = solve(q_y2, y)
print("Roots y3 = {}, {}".format(roots_y3[0].evalf(), roots_y3[1].evalf()))
print("Roots y4 = {}, {}".format(roots_y4[0].evalf(), roots_y4[1].evalf()))
