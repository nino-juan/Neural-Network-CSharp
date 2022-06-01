//using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class net
{
    public List<Layer> m_layers = new List<Layer> { };

    float m_error = 0;
    float m_recentAveragedError;
    float m_recentAveragedSmoothingFactor;

    public net(int[] topology)
    {
        int numLayers = topology.Length;

        for (int layerNum = 0; layerNum < numLayers; layerNum++)
        {
            int neuronOutputs = 0;
            if (layerNum < numLayers - 1)
                neuronOutputs = topology[layerNum + 1];
            //create a new layer and add to the m_layer container
            m_layers.Add(new Layer());
            //bias neuron added with <=
            for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
            {
                m_layers[layerNum].neuronas.Add(new Neuron(neuronOutputs, neuronNum)); //add a new neuron to the layer container of neurons
            }

            //force last neuron (bias) to output 1
            m_layers[layerNum].neuronas[topology[layerNum]].SetOutputValue(1.0f);
        }

        return;
    }

    public void FeedForward(float[] inputValues)
    {
        int inputNeurons = m_layers[0].neuronas.Count;
        if (inputValues.Length != inputNeurons - 1) //-1 for the bias neuron
            Debug.LogError("net.FeedForward:Input number is not equal to neurons on input layer");
        //inputs to first layer
        for (int i = 0; i < inputNeurons - 1; i++)
        {
            m_layers[0].neuronas[i].SetOutputValue(inputValues[i]);
        }

        //forward propagation
        for (int layerNum = 1; layerNum < m_layers.Count; layerNum++)
        { //from the first hidden layer on
            Layer previousLayer = m_layers[layerNum - 1]; //set the new output for each neuron in this layer
            for (int n = 0; n < m_layers[layerNum].neuronas.Count - 1; n++)
            { //except for bias, cause bias output value is always 1
                m_layers[layerNum].neuronas[n].FeedForward(previousLayer); //set the new output value for this neuron depending on the incomming signals
            }
        }
    }

    public void BackProp(float[] targetValues)
    {
        //Calculate overall net error (RMS)

        Layer output_layer = m_layers[m_layers.Count - 1];
        m_error = 0;
        int numNeuronas = output_layer.neuronas.Count - 1; //without bias

        for (int n = 0; n < numNeuronas; n++)
        {
            float delta = targetValues[n] - output_layer.neuronas[n].GetOutputValue();
            m_error += delta * delta;
        } //check the error of the network
        m_error /= numNeuronas;
        m_error = Mathf.Sqrt(m_error);

        //Implement a recent average measurement
        //m_recentAveragedError = (m_recentAveragedError * m_recentAveragedSmoothingFactor + m_error)
        /// (m_recentAveragedSmoothingFactor * 1.0f);

        // Calculate output layer gradientss
        for (int n = 0; n < numNeuronas; n++)
        {
            output_layer.neuronas[n].CalcOutputGradients(targetValues[n]);
        }
        //Calculate gradients on hidden layers

        for (int layerNum = m_layers.Count - 2; layerNum > 0; layerNum--)
        { //DEBUG CHECK IF layerNum >0 , o layerNum>=0
            Layer hiddenLayer = m_layers[layerNum];
            Layer nextLayer = m_layers[layerNum + 1];
            for (int n = 0; n < hiddenLayer.neuronas.Count; n++)
            {
                hiddenLayer.neuronas[n].CalcHiddenGradients(nextLayer);
            }
        }

        //For all layers from output to first hidden layer,
        //update connection weights

        for (int layerNum = m_layers.Count - 1; layerNum > 0; layerNum--)
        {
            Layer layer = m_layers[layerNum];
            Layer previousLayer = m_layers[layerNum - 1];

            for (int n = 0; n < layer.neuronas.Count - 1; n++)
            {
                layer.neuronas[n].UpdateInputWeights(previousLayer);
            }
        }
    }

    public void GetResults(out float[] results)
    {
        int outputNum = m_layers[m_layers.Count - 1].neuronas.Count - 1;
        results = new float[outputNum];

        for (int n = 0; n < outputNum; n++)
        {
            results[n] = m_layers[m_layers.Count - 1].neuronas[n].GetOutputValue();
        }
    }

    public string ExportToCSV()
    {
        string topology = "";
        for (int layerNum = 0; layerNum < m_layers.Count; layerNum++)
        { // for each layer, add the number of neurons in topollogy string
            topology += m_layers[layerNum].neuronas.Count - 1; //neuron num(-bias) + separator
            //Debug.Log("numero de layers");
            if (layerNum < m_layers.Count - 1)
            {
                topology += ",";
            }
        }
        topology += "\n";
        string neurons = "";
        for (int layerNum = 0; layerNum < m_layers.Count; layerNum++)
        { // for each layer save the neurons weights in string
            //for each neuron on this layer, including bias, add a row of values
            for (int neuron = 0; neuron < m_layers[layerNum].neuronas.Count; neuron++)
            {
                Neuron currentNeuron = m_layers[layerNum].neuronas[neuron]; // get current neuron

                string weightsString = "";
                for (int weightNum = 0; weightNum < currentNeuron.neuronWeights.Count; weightNum++)
                { // all the neuron weights in this neuron
                    weightsString += currentNeuron.neuronWeights[weightNum].weight;
                    if (weightNum < currentNeuron.neuronWeights.Count - 1)
                    { //add separator, unless is the last weight
                        weightsString += ",";
                    }
                }
                neurons += weightsString + "\n"; // append the line of neuron weights to string//nextline = next neuron
            }
        }

        return topology + neurons;
    }

    public void ImportWeightsFromCSV(string WeightsCVSstring)
    {
        //	Debug.LogWarning (WeightsCVSstring);
        ///Should receive a string in the form of "w1,1,w1,2,...w1,n \n w2,1,w2,2,...w2,n \n .... wm,1,wm,2,...wm,n \n "

        string[] lines = WeightsCVSstring.Split('\n');
        int neuronCount = 0;

        for (int layer = 0; layer < m_layers.Count - 1; layer++)
        { //every layer except for the output
            for (int n = 0; n < m_layers[layer].neuronas.Count; n++)
            { // for each neuron in the layer
                //current neuron
                Neuron currentNeuron = m_layers[layer].neuronas[n]; // current neuron

                currentNeuron.LoadCSVWeights(lines[neuronCount]);
                neuronCount++;
            }
        }

        return;
    }
}

public class Neuron
{
    //learning variables


    float m_neuronOutput,
        m_gradient,
        eta = 0.2f,
        alpha = 0.5f;
    int neuronIndex;

    public List<Connections> neuronWeights = new List<Connections> { };

    public Neuron(int outputNum, int m_neuronIndex)
    {
        neuronIndex = m_neuronIndex;

        for (int connect = 0; connect < outputNum; connect++)
        {
            neuronWeights.Add(new Connections());
            neuronWeights[connect].weight = RandomWeight();
        }

        //Debug.Log ("Neurona Creada!");
    }

    float RandomWeight()
    {
        return Random.value;
    }

    public void SetOutputValue(float value)
    {
        m_neuronOutput = value;
    }

    public float GetOutputValue()
    {
        return m_neuronOutput;
    }

    public void FeedForward(Layer previousLayer)
    {
        float sum = 0;
        //weightsum the inputs of the last layer, including bias
        for (int n = 0; n < previousLayer.neuronas.Count; n++)
        {
            sum +=
                previousLayer.neuronas[n].GetOutputValue()
                * previousLayer.neuronas[n].neuronWeights[neuronIndex].weight;
        }

        m_neuronOutput = TransferFunction(sum);
    }

    public void CalcOutputGradients(float targetValue)
    {
        float delta = targetValue - m_neuronOutput;
        m_gradient = delta * TransferFunctionDerivative(m_neuronOutput);
    }

    public void CalcHiddenGradients(Layer nextLayer)
    {
        //derivative of weights of the next layer
        float dow = sumDOW(nextLayer);

        m_gradient = dow * TransferFunctionDerivative(m_neuronOutput);
    }

    float TransferFunction(float sum)
    {
        //sigmoid function
        return 1 / (1 + Mathf.Exp(-sum));
    }

    float TransferFunctionDerivative(float sum)
    {
        float y = TransferFunction(sum);
        return (y * (1 - y));
    }

    public void LoadCSVWeights(string CSVWeights)
    {
        ///should receive a string in the form of "weight1,weight2,...,weightn"
        string[] weightString = CSVWeights.Split(','); // separate individual weights

        for (int w = 0; w < weightString.Length; w++)
        { //for every weight assign the weight to the network
            float readWeight = 0;
            if (!float.TryParse(weightString[w], out readWeight))
            { // try to convert the text to an float number
                Debug.LogError("LoadCSVWeight: Incorrect neuron weight structure!");
                return;
            }

            neuronWeights[w].weight = readWeight; //set the read value to the weight number
            //	Debug.Log("This weight is read: "+readWeight+" weight number: "+ w);
        }
    }

    public void UpdateInputWeights(Layer previousLayer)
    {
        //the weights to be updated are in the connection container
        //in the neurons in the preceding layer
        int numNeuronas = previousLayer.neuronas.Count;
        for (int n = 0; n < numNeuronas; n++)
        { //for every previous layer neuron
            Neuron neu = previousLayer.neuronas[n]; //getting the previous layer neuron
            float oldDeltaW = neu.neuronWeights[neuronIndex].deltaWeight;
            float deltaWeight =
                //Individual input magnified by the gradient and trainrate
                eta * neu.GetOutputValue() * m_gradient
                //also add momentum = a fraction of the previos delta weight
                + alpha * oldDeltaW;

            neu.neuronWeights[neuronIndex].deltaWeight = deltaWeight;
            neu.neuronWeights[neuronIndex].weight += deltaWeight;
            //	float debug = neu.neuronWeights [neuronIndex].weight;
            //	Debug.Log("updated neuron "+neuronIndex +" deltaweight to: " +deltaWeight+ " and weight is now: " + debug );
        }
    }

    public float sumDOW(Layer nextLayer)
    {
        float sum = 0;
        // sum our contributions of the errors at the nodes we feed
        for (int ni = 0; ni < nextLayer.neuronas.Count - 1; ni++)
        {
            sum += neuronWeights[ni].weight * nextLayer.neuronas[ni].m_gradient;
        }
        return sum;
    }
}

public class Layer
{
    public List<Neuron> neuronas = new List<Neuron> { };
}

public class Connections
{
    public float weight,
        deltaWeight;
}
