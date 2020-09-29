import numpy as np
class NN:
    def __init__(self,x,y):
        self.w1 = np.random.rand(x.shape[1], 3)
        self.w2= np.random.rand(3,1)
        self.a2 = np.zeros(y.shape)
    def sigmoid(self,z):
        return 1.0 / (1 + np.exp(-z))
    def sigmoiddertive(self,z):
        return z * (1.0 - z)
    def forword(self,x):
        self.z1 = np.dot(x ,self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1,self.w2)
        self.a2 =  self.sigmoid(self.z2)
 
    def backprop(self,y,x):
        self.o = y - self.a2
        self.o_delta = self.o * self.sigmoiddertive(self.o)
 
        self.a1_d = np.dot(self.o_delta,self.w2.T )
        self.a1_delta = self.a1_d * self.sigmoiddertive(self.w2)
        
        self.w1 += np.dot(x.T, self.a1_delta )
        self.w2 += np.dot(self.a1.T, self.o_delta)
 
    def predict(self): ##teeeest
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forword(xPredicted)))
    
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

xPredicted = np.array(([4,8]), dtype=float)

# x = x/np.amax(x, axis=0) 
# xPredicted = xPredicted/np.amax(xPredicted, axis=0)  
# y = y/100  


n = NN(x,y)
for i in range(1000):
    n.forword(x)
    n.backprop(y,x)
print(n.a2)     
# print(n.w1)    
n.predict()