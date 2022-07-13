import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')

# just to see your loss won't use
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        total_error += (y - (m*x+b))**2
    total_error/float(len(points))

# let L be the learning rate
# making a function using the the function which was a derivative of Error func wrt m and b
def gradient_descent(m_now, b_now, points , L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        m_gradient += -(2/n) * x * (y - (m_now*x+b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m,b

m = 0
b = 0
#learningRate = 0.0000001  For SAT-GPA dataset
learningRate = 0.001
epochs = 1000

for i in range(epochs):
    m,b = gradient_descent(m, b, data, learningRate)


plt.scatter(data.X, data.Y, color = "black")
#Diffrent for diff dataset Line:42
plt.plot(list(range(1,20)),[m*x + b for x in range(1,20)],color = "red")
plt.show()