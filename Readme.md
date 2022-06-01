# net.cs

My own C# implementation of Dave Miller's C Neural Network class, which I coded following its awesome [C++ explanation](https://www.youtube.com/watch?v=KkwX7FkLfug).

I wrote the code for a previous project where I had to build and train a neural network to classify gestures from accelerometer data. So I implemented these functions

- `public net(int[] topology)` Sets up a new neural network based on an array describing the number of neurons on each layer.
- `public void FeedForward(float[] inputValues)` Calculates the output values after feeding the network with an array of input values.
- `public void BackProp(float[] targetValues)` A backpropagation training algorithm, considering the momentum.

I also added code to save and load the ANN topology and weights to and from a CSV file.

- `public string ExportToCSV()` Turns the ANN topology and connection weights into a CSV string.

- `public void ImportWeightsFromCSV(string WeightsCVSstring)` Loads the weight values for the ANN from a CSV string. Note that it should receive a string in the form of:

```
w1,1,w1,2,...w1,n \n w2,1,w2,2,...w2,n \n .... wm,1,wm,2,...wm,n \n
```

Although its been a good while since I updated the code, I plan on cleaning up the code and adding a demo in the close future.
