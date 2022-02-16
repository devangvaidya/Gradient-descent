from sympy import *
from decimal import *
import matplotlib.pyplot as plt
z1 = Symbol('z1')
z2 = Symbol('z2')
a = 2
b = 90
f = (a-z1)**2 + (b*(z2 - z1**2)**2)
f_z1 = -2*(a-z1) - (4*z1*b*(z2 -z1**2))
f_z2 = 2*b*(z2-z1**2)



x=theta1_current = 4  #x
y=theta2_current = 9 #y
learning_rate = 0.0001
iterations = 0
precision = 0.00001
maxIterations = 100000
funtion_list = []
iteration_list = []

def evaluate_z1(theta1_current,theta2_current):
    return f_z1.subs(z1,theta1_current).subs(z2,theta2_current)

def evaluate_z2(theta1_current,theta2_current):
    return f_z2.subs(z1,theta1_current).subs(z2,theta2_current) 

def loss_value(theta1_current,theta2_current):
    return f.subs(z1,theta1_current).subs(z2,theta2_current)   

while True:
    theta1_next = theta1_current - learning_rate*(evaluate_z1(theta1_current,theta2_current))
    theta2_next = theta2_current - learning_rate*(evaluate_z2(theta1_current,theta2_current))

    
    iterations += 1
    if iterations > maxIterations:
        print("Iteration number: {} z1: {} and z2: {}".format(iterations,theta1_next,theta2_next))
        print("Too many iterations.")
        break

    #If the value of theta changes less of a certain amount, our goal is met.
    if abs(theta1_next-theta1_current) < precision and abs(theta2_next-theta2_current) < precision:
        print("Iteration number: {} z1: {} and z2: {}".format(iterations,theta1_next,theta2_next))
        break

    #Simultaneous update
    theta1_current = theta1_next
    theta2_current = theta2_next

    if (iterations % 1000 ==0):
        loss = loss_value(theta1_next,theta2_next)
        funtion_list.append(loss)
        iteration_list.append(iterations)
        print("Iteration number: {} Loss: {} z1: {} and z2: {}".format(iterations,loss,theta1_next,theta2_next))

plt.plot(iteration_list,funtion_list)
plt.xlabel("Number Of Iterations")
plt.ylabel("Function g(z1,z2)")
plt.title("a={} b={} z1={} z2={} learning rate={}".format(a,b,x,y,learning_rate))
plt.show()