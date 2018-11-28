from pylab import *

def HenonMap(a,b,x,y):
	return y + 1.0 - a *x*x, b * x

# Map dependent parameters
a = 1.4
b = 0.3
iterates = 100

# Initial Condition
xtemp = 0.1
ytemp = 0.3

x = [xtemp]
y = [ytemp]

for n in range(0,iterates):
  xtemp, ytemp = HenonMap(a,b,xtemp,ytemp)
  x.append( xtemp )
  y.append( ytemp )
  print(xtemp, "   ", ytemp)

# Plot the time series
plot(x,y, 'b,')
show()
