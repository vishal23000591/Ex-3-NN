<H3>Name: Vishal S </H3>
<H3>Register No: 212223110063 </H3>
<H3>EX. NO.3</H3>
<H3>DATE:</H3>

<H2 aligh = center> Implementation of MLP for a non-linearly separable data</H2>

<h3>Aim:</h3>
To implement a perceptron for classification using Python

<H3>Theory:</H3>a
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

<h3>Algorithm :</H3>

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

# Program:
1.importing packages:
```
import numpy as np
import matplotlib.pyplot as plt
```
2.model initialization:
```
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

input_size=2
hidden_layer=3
output_size=1

w1=np.random.randn(input_size,hidden_layer)
b1=np.zeros((1,hidden_layer))
w2=np.random.randn(hidden_layer,output_size)
b2=np.zeros((1,output_size))

def sigmoid_function(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  return x*(1-x)
```
3.model mechanism:
```
losses=[]

for epochs in range(10000):
  hidden_input=np.dot(x,w1)+b1
  hidden_output=sigmoid_function(hidden_input)
  output_layer_input=np.dot(hidden_output,w2)+b2
  output=sigmoid_function(output_layer_input)

  error=y-output
  loss=np.mean(error**2)
  losses.append(loss)

  d_out=error*sigmoid_derivative(output)
  d_hidden=np.dot(d_out, w2.T)*sigmoid_derivative(hidden_output)

  w2 += np.dot(hidden_output.T, d_out) * 0.1
  b2 += np.sum(d_out, axis=0, keepdims=True) * 0.1

  w1 += np.dot(x.T, d_hidden) * 0.1
  b1 += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

for v,out in zip(x,output):
  print(f"Input: {v}, Output: {np.round(out)}")
```
4.plotting:
```
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```
# Output:
![image](https://github.com/user-attachments/assets/6a1732d9-a4dd-4f65-9036-24a0c3dad0cf)
![image](https://github.com/user-attachments/assets/d934596f-e37a-4f6a-9d86-ce98148ad086)

<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 
