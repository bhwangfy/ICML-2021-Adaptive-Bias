import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from mpl_toolkits.mplot3d import Axes3D

def sigma(x):
    if x >= 0:
        return x
    else:
        return x / 2


def sigma_derivative(x):
    if x >= 0:
        return 1.0
    else:
        return 1.0 / 2


def inner_product(x, y):
    return x[0] * y[0] + x[1] * y[1]


def gradient_cal(x,y, w, v):
    gradient_1=-math.exp(-v * y * sigma(inner_product(w, x))) * y * v * sigma_derivative(inner_product(w, x)) * x[0]
    gradient_2 = -math.exp(-v * y * sigma(inner_product(w, x))) * y * v * sigma_derivative(
            inner_product(w, x)) * x[1]
    gradient_3= -math.exp(-y * v * sigma(inner_product(w, x))) * y * sigma(inner_product(w, x))
    return gradient_1, gradient_2, gradient_3

def prediction_cal(x, w, v):
    return v * sigma(inner_product(w, x))

def GD(w_input, v_input, x_input, y_input):
    w = deepcopy(w_input)
    v = deepcopy(v_input)
    x = deepcopy(x_input)
    y= deepcopy(y_input)
    lr=0.01
    training_accuracy=[]
    print('GD')
    for i in range(5000):
        gradient = [0, 0, 0]
        for j in range(len(x)):
            gradient_current=gradient_cal(x[j],y[j],w,v)
            gradient[0] += gradient_current[0]
            gradient[1] += gradient_current[1]
            gradient[2] += gradient_current[2]
        gradient[0] /= len(x)
        gradient[1] /= len(x)
        gradient[2] /= len(x)
        w[0] -= lr* gradient[0]
        w[1] -=lr* gradient[1]
        v -= lr* gradient[2]
        cal=0
        for j in range(len(x)):
            if y[j]*prediction_cal(x[j],w,v)>0:
                cal += 1
        training_accuracy.append(cal*100/len(x))
    normalize = math.sqrt(w[0] ** 2 + w[1] ** 2 + v ** 2)
    min=v * y[0] * sigma(inner_product(w, x[0]))/normalize**2
    for j in range(1,len(x)):
        if min>v * y[j] * sigma(inner_product(w, x[j]))/normalize**2:
            min=v * y[j] * sigma(inner_product(w, x[j]))/normalize**2
    print(min)
    return [w[0] / normalize, w[1] / normalize, v / normalize], min, training_accuracy


def AdaGrad(w_input, v_input, x_input, y_input):
    w = deepcopy(w_input)
    v = deepcopy(v_input)
    x = deepcopy(x_input)
    y = deepcopy(y_input)
    adaptor = [10 ** (-5), 10 ** (-5), 10 ** (-5)]
    update=[0,0,0]
    lr=0.1
    training_accuracy=[]
    print('ada')
    for i in range(5000):
        gradient = [0, 0, 0]
        for j in range(len(x)):
            gradient_current = gradient_cal(x[j], y[j], w, v)
            gradient[0] += gradient_current[0]
            gradient[1] += gradient_current[1]
            gradient[2] += gradient_current[2]
        gradient[0] /= len(x)
        gradient[1] /= len(x)
        gradient[2] /= len(x)
        adaptor[0] += gradient[0] ** 2
        adaptor[1] += gradient[1] ** 2
        adaptor[2] += gradient[2] ** 2
        update[0]= gradient[0] / math.sqrt(adaptor[0])
        update[1]=  gradient[1] / math.sqrt(adaptor[1])
        update[2] = gradient[2] / math.sqrt(adaptor[2])
        w[0] -=lr* update[0]
        w[1] -=lr* update[1]
        v -=lr* update[2]
        cal = 0
        for j in range(len(x)):
            if y[j] * prediction_cal(x[j], w, v) > 0:
                cal += 1
        training_accuracy.append(cal*100 / len(x))
    normalize2 = math.sqrt(math.sqrt(adaptor[0])+math.sqrt(adaptor[1])+math.sqrt(adaptor[2]))
    normalize = math.sqrt(w[0] ** 2 + w[1] ** 2 + v ** 2)
    min = v * y[0] * sigma(inner_product(w, x[0])) / normalize ** 2
    for j in range(1, len(x)):
        if min > v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2:
            min = v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2
    print(min)
    return [math.sqrt(math.sqrt(adaptor[0]) )/ normalize2, math.sqrt(math.sqrt(adaptor[1]))/ normalize2,
            math.sqrt(math.sqrt(adaptor[2])) / normalize2], min,training_accuracy


def RMSPROP (w_input, v_input, x_input, y_input, beta):
    w = deepcopy(w_input)
    v = deepcopy(v_input)
    x = deepcopy(x_input)
    y = deepcopy(y_input)
    adaptor = [0, 0, 0]
    update=[0,0,0]
    training_accuracy=[]
    lr=0.1
    print('rms')
    for i in range(5000):
        gradient = [0, 0, 0]
        for j in range(len(x)):
            gradient_current = gradient_cal(x[j], y[j], w, v)
            gradient[0] += gradient_current[0]
            gradient[1] += gradient_current[1]
            gradient[2] += gradient_current[2]
        gradient[0] /= len(x)
        gradient[1] /= len(x)
        gradient[2] /= len(x)
        adaptor[0] = (1 - beta) * adaptor[0] + beta * gradient[0] ** 2
        adaptor[1] = (1 - beta) * adaptor[1] + beta * gradient[1] ** 2
        adaptor[2] = (1 - beta) * adaptor[2] + beta * gradient[2] ** 2
        update[0]= gradient[0] / (math.sqrt(adaptor[0]+ 10 ** (-5)) )
        update[1] = gradient[1] / (math.sqrt(adaptor[1] + 10 ** (-5)))
        update[2] = gradient[2] / (math.sqrt(adaptor[2] + 10 ** (-5)))
        w[0] -=lr* update[0]
        w[1] -=lr* update[1]
        v -=lr* update[2]
        cal = 0
        for j in range(len(x)):
            if y[j] * prediction_cal(x[j], w, v) > 0:
                cal += 1
        training_accuracy.append(cal*100 / len(x))
    normalize2 = math.sqrt(math.sqrt(adaptor[0]+ 10 ** (-5))+math.sqrt(adaptor[1]+ 10 ** (-5))
                           +math.sqrt(adaptor[2]+ 10 ** (-5)))
    normalize = math.sqrt(w[0] ** 2 + w[1] ** 2 + v ** 2)
    min = v * y[0] * sigma(inner_product(w, x[0])) / normalize ** 2
    for j in range(1, len(x)):
        if min > v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2:
            min = v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2
    print(min)
    return [math.sqrt(math.sqrt(adaptor[0]+ 10 ** (-5))) / normalize2, math.sqrt(math.sqrt(adaptor[1]+ 10 ** (-5))) / normalize2,
            math.sqrt(math.sqrt(adaptor[2]+ 10 ** (-5))) / normalize2], min,training_accuracy

def Adam (w_input, v_input, x_input, y_input, beta):
    w = deepcopy(w_input)
    v = deepcopy(v_input)
    x = deepcopy(x_input)
    y = deepcopy(y_input)
    update=[0,0,0]
    adaptor = [0, 0, 0]
    lr=0.1
    training_accuracy=[]
    print('adam')
    for i in range(5000):
        gradient = [0, 0, 0]
        for j in range(len(x)):
            gradient_current = gradient_cal(x[j], y[j], w, v)
            gradient[0] += gradient_current[0]
            gradient[1] += gradient_current[1]
            gradient[2] += gradient_current[2]
        gradient[0] /= len(x)
        gradient[1] /= len(x)
        gradient[2] /= len(x)
        adaptor[0] = (1 - beta) * adaptor[0] + beta * gradient[0] ** 2
        adaptor[1] = (1 - beta) * adaptor[1] + beta * gradient[1] ** 2
        adaptor[2] = (1 - beta) * adaptor[2] + beta * gradient[2] ** 2
        update[0] = gradient[0] / (math.sqrt(adaptor[0]/(1-beta**(i+1)) + 10 ** (-5)))
        update[1] = gradient[1] / (math.sqrt(adaptor[1]/(1-beta**(i+1)) + 10 ** (-5)))
        update[2] = gradient[2] / (math.sqrt(adaptor[2]/(1-beta**(i+1)) + 10 ** (-5)))
        w[0] -= lr * update[0]
        w[1] -= lr * update[1]
        v -= lr * update[2]
        cal = 0
        for j in range(len(x)):
            if y[j] * prediction_cal(x[j], w, v) > 0:
                cal += 1
        training_accuracy.append(cal*100 / len(x))
    normalize2 = math.sqrt(math.sqrt(adaptor[0] + 10 ** (-5)) + math.sqrt(adaptor[1] + 10 ** (-5))
                           + math.sqrt(adaptor[2] + 10 ** (-5)))
    normalize = math.sqrt(w[0] ** 2 + w[1] ** 2 + v ** 2)
    min = v * y[0] * sigma(inner_product(w, x[0])) / normalize ** 2
    for j in range(1, len(x)):
        if min > v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2:
            min = v * y[j] * sigma(inner_product(w, x[j])) / normalize ** 2
    print(min)
    return [math.sqrt(math.sqrt(adaptor[0] + 10 ** (-5))) / normalize2,
            math.sqrt(math.sqrt(adaptor[1] + 10 ** (-5))) / normalize2,
            math.sqrt(math.sqrt(adaptor[2] + 10 ** (-5))) / normalize2], min,training_accuracy

x = []
x_positive=[]
x_negative=[]
y = []
origin_x=[]
origin=[]
#create_data_set
for i in range (0,50):
    u=[math.cos(0.5)+random.uniform(-0.6,0.6), math.sin(0.5)+random.uniform(-0.6,0.6)]
    x.append(u)
    x_positive.append(u)
    y.append(1)
    origin_x.append(0)
    v=[-math.cos(0.5) + random.uniform(-0.6, 0.6), -math.sin(0.5) + random.uniform(-0.6, 0.6)]
    x.append(v)
    x_negative.append(v)
    y.append(-1)
origin_x_2 = deepcopy(origin_x)
origin.append(origin_x)
origin.append(origin_x_2)
origin_2=deepcopy(origin)
x_positive_array=np.array(x_positive)
x_negative_array=np.array(x_negative)
origin_array=np.array(origin)
origin_2_array=np.array(origin_2)

fig, ax_0 = plt.subplots()
q1=ax_0.quiver(origin_array[:,0],origin_array[:,1], x_positive_array[:,0], x_positive_array[:,1],units='xy' ,scale=1, color = 'red')
q2=ax_0.quiver(origin_2_array[:,0],origin_2_array[:,1], x_negative_array[:,0], x_negative_array[:,1], units='xy' ,scale=1,color = 'green')
ax_0.set_aspect('equal')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

num_item=100
round=100
soa=[]
sor=[]
som=[]
array_ada=[]
array_rms=[]
array_gd=[]
array_adam=[]
rms_training_store=[]
ada_training_store=[]
gd_training_store=[]
adam_training_store=[]
for j in range(5000):
    rms_training_store.append(0)
    gd_training_store.append(0)
    ada_training_store.append(0)
    adam_training_store.append(0)
sos=[np.array(([ 0,0,0, 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)]))]
for i in range(200):
#    x_2 = [-2 * math.cos(0.5) + random.uniform(-0.4, 0.4), -2 * math.sin(0.5) + random.uniform(-0.4, 0.4)]
    w = [random.uniform(-1,1),random.uniform(-1,1)]
    v = random.uniform(-1,1)
    rms, margin_rms,rms_training= RMSPROP(w, v, x,y, 0.9)
    ada, margin_ada,ada_training = AdaGrad(w, v, x, y)
    gd, margin_gd, gd_training = GD(w, v, x, y)
    adam, margin_adam, adam_training = Adam(w, v, x, y, 0.9)
    if margin_rms<0 or margin_ada<0 or margin_adam<0 or margin_gd<0:
        continue
    array_ada.append(margin_ada)
    array_rms.append(margin_rms)
    array_gd.append(margin_gd)
    array_adam.append(margin_adam)
    #print(ada[0]/math.sqrt(ada[0]**2+ada[1]**2),ada[1]/math.sqrt(ada[0]**2+ada[1]**2))
    soa.append( np.array([0, 0, 0, ada[0], ada[1], ada[2]]))
    sor.append( np.array([0, 0, 0, rms[0], rms[1], rms[2]]))
    som.append( np.array([0, 0, 0, adam[0], adam[1], adam[2]]))
    for j in range(5000):
        rms_training_store[j]+=rms_training[j]
        gd_training_store[j]+=gd_training[j]
        ada_training_store[j]+=ada_training[j]
        adam_training_store[j]+=adam_training[j]
    if len(array_ada)==num_item:
        break
for j in range(5000):
    rms_training_store[j]/=num_item
    gd_training_store[j]/=num_item
    ada_training_store[j]/=num_item
    adam_training_store[j]/=num_item
X_a, Y_a, Z_a, U_a, W_a, V_a = zip(*soa)
X_r, Y_r, Z_r, U_r, W_r, V_r = zip(*sor)
X_s, Y_s, Z_s, U_s, W_s, V_s = zip(*sos)
X_m, Y_m, Z_m, U_m, W_m, V_m = zip(*som)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X_a, Y_a, Z_a, U_a, W_a, V_a, color = 'red')
ax.quiver(X_s, Y_s, Z_s, U_s, W_s, V_s, color = 'green')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.quiver(X_r, Y_r, Z_r, U_r, W_r, V_r, color = 'blue')
ax2.quiver(X_s, Y_s, Z_s, U_s, W_s, V_s, color = 'green')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_zlim([0, 1])
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.quiver(X_m, Y_m, Z_m, U_m, W_m, V_m, color = 'black')
ax3.quiver(X_s, Y_s, Z_s, U_s, W_s, V_s, color = 'green')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.set_zlim([0, 1])
plt.figure()
A=[]
for i in range(len(array_gd)):
    A.append(i)
l_sgd,=plt.plot(A,array_gd,linestyle='-')
l_rmsprop,=plt.plot(A,array_rms,linestyle='-')
l_adagrad,=plt.plot(A,array_ada,linestyle='-')
l_adam,=plt.plot(A,array_adam,linestyle='-')
plt.xlabel("Round")#x轴上的名字
plt.ylabel("Margin")
plt.legend(handles=[l_sgd,l_rmsprop,l_adagrad,l_adam],labels=['SGD','RMSPROP','AdaGrad','Adam'])
# plt.legend(handles=[l_sgd,l_rmsprop,l_adagrad],labels=['SGDM','RMSPROP','AdaGrad'])
plt.figure()
B=[]
for i in range(len(gd_training_store)):
    B.append(i)
l_sgd,=plt.plot(B,gd_training_store,linestyle='-')
l_rmsprop,=plt.plot(B,rms_training_store,linestyle='-')
l_adagrad,=plt.plot(B,ada_training_store,linestyle='-')
l_adam,=plt.plot(B,adam_training_store,linestyle='-')
plt.xlabel("Round")#x轴上的名字
plt.ylabel("Margin")
plt.legend(handles=[l_sgd,l_rmsprop,l_adagrad,l_adam],labels=['SGD','RMSPROP','AdaGrad','Adam'])
plt.show()
