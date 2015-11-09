#include "NeuralNet.h"

/*
Neural Network Layer class
*/
NNLayer::NNLayer(int units, int inputs)
{
  no_units = units;   no_inputs = inputs;
  
  bias = new float[no_units];
  output = new float[no_units];
  delta = new float[no_units];
  input = new float[no_units];
  weights = new float[no_inputs * no_units];
  
  prev = NULL; next = NULL;
  
  cerr << "NNLayer : Created a layer with " << no_units << " units and " << no_inputs << " inputs\n"; 
}

void NNLayer::randomiseWeights()
{
  srand ( time(NULL) );
  
  // Randomise the input weights
  for(int i = 0; i < (no_inputs * no_units); i++)
    weights[i] = (rand() / (float)RAND_MAX) * 2 - 1;

  // Randomise the biases 
  for(int i = 0; i < no_inputs; i++)
    bias[i] = (rand() / (float)RAND_MAX) * 2 - 1;
}


// getWeight where i = input, o = output unit
float NNLayer::getWeight(int i, int o) {
  return weights[(o - 1) * no_inputs + (i - 1)];
}

// setWeight where i = input, o = output unit, w = weight
void NNLayer::setWeight(int i, int o, float w) {
  weights[(o - 1) * no_inputs + (i - 1)] = w;
}

// updateWeight where i = input, o = output unit, wChange = weight change
void NNLayer::updateWeight(int i, int o, float wChange) {
  weights[(o - 1) * no_inputs + (i - 1)] = weights[(o - 1) * no_inputs + (i - 1)] + wChange;
}

// Log-Sigmoid function
float NNLayer::logSig(const float & x) {
  return 1 / ( 1 + (float)exp(-x)); 
}

// Log-Sigmoid derivative function
float NNLayer::logSigDeriv(const float & x) {
  return x * ( 1 - x); 
}

// Recursive forward propagation function
void NNLayer::forwardProp(float * inp)
{
  float wSumInput;
  
  // Keep a record of the input for back propagation as I don't store an input layer
  for(int i = 0; i < no_inputs; i++)
    input[i] = inp[i];
  
  // Go through all the units
  for(int u = 0; u < no_units; u++) {
    wSumInput = 0;
    // Go through all the inputs
    for(int i = 0; i < no_inputs; i++)
      wSumInput += inp[i] * getWeight(i + 1, u + 1);
    
    // Add the bias
    wSumInput += bias[u];
    
    // Calculate the output using the log-sigmoid function
    output[u] = logSig(wSumInput);
    if (output[u] != output[u]) { cerr << "ERROR: NaN output detected\n"; exit(0); } 
  }

  // Recursively propagate the output as the input to the next layer
  if(next) { next->forwardProp(output); }
}



// Recursive back propagation function
void NNLayer::backProp(float * prevDelta, const float & learnRate) {
  // Check if it's the output layer or hidden layer and calculate the error accordingly
  if(!next) {
    // Output layer
    for(int u = 0; u < no_units; u++) {
      delta[u] = (prevDelta[u] - output[u]) * logSigDeriv(output[u]);
      if (delta[u] != delta[u]) { cout << "Fuck a NaN"; exit(0); }
    }
  }
  else if(next) {
    // Hidden layer, find out the number of nodes in the next layer
    int unitsNext = next->getNoUnits(); 
    for(int u = 0; u < no_units; u++) { 
      float temp = 0;
      for(int o = 0; o < unitsNext; o++)
        temp += next->getWeight(u + 1, o + 1) * prevDelta[o];
      delta[u] = logSigDeriv(output[u]) * temp;
    }   
  }
  
  // Recursively propagate the error to the prev layer
  if(prev) prev->backProp(delta, learnRate);
  
  // For each unit update the biases and the weights connected to it
  for(int u = 0; u < no_units; u++) {
    // Update the biases
    bias[u] = bias[u] + (learnRate * delta[u] * 1); 

    // Update the weights
    for(int i = 0; i < no_inputs; i++) {
      updateWeight(i + 1, u + 1, learnRate * input[i] * delta[u]);
    }
  }
}



/*
Neural Network class
*/
NeuralNet::NeuralNet(int no_inputs, int no_hidden, int no_output)
{
  // Create the first / hidden layer
  first = new NNLayer(no_hidden, no_inputs);
  // Create the second / output layer
  last = new NNLayer(no_output, no_hidden);
  // Have pointers to the first and last layers
  first->setNext(last); last->setPrev(first);
  
  cerr << "NeuralNet: Created with " << no_inputs << " inputs, " << no_hidden << " hidden and " << no_output << " outputs\n";
}


void NeuralNet::randomiseWeights() {
  first->randomiseWeights();
  last->randomiseWeights();
}


// Train the Neural Network using this dataset
stats NeuralNet::train(vector<rowdata*> & dataset) {
  int correct, total, epochs = 0;
  vector<rowdata*>::iterator i;
  float * output, * target = new float[2];
  rowdata * example;
  bool stop = false;
  cerr << "NeuralNet: Training the network with " << dataset.size() << " examples..."<< endl;
  
  // Loop through atleast one epoch then check stopping criteria 
  do {
    i = dataset.begin();
    
    // Go through all the examples and update the weights accordingly     
    while(i != dataset.end()) {
      // Get the example, forward propagate it
      example = (*i);
      first->forwardProp(example->attributes);
      
      // Binary encode the target input, only two classes supported at this stage
      if(example->target == 0) { target[0] = 1; target[1] = 0;}
      else if(example->target == 1) { target[0] = 0; target[1] = 1;}
      
      // Pass the output layer the target value, it will calculate the error and backpropagate it
      last->backProp(target, learnRate);      
      i++;    
    }

    // Now check all the examples again, calculate the accuracy and mean square error 
    i = dataset.begin(); total = 0; correct = 0;
    float totSqrError = 0;
    while(i != dataset.end()) {
      // Forward propagate the input, which will propagate its output to the next layer and so on
      example = (*i);
      first->forwardProp(example->attributes);
      
      // Get the resulting output, check if it was classifed correctly
      output = last->getOutput();
      if(example->target == 0 && output[0] > output[1]) correct++;
      else if(example->target == 1 && output[0] < output[1]) correct++;
      
      // Binary encode the target input, only two classes support at this stage
      if(example->target == 0) { target[0] = 1; target[1] = 0;}
      else if(example->target == 1) { target[0] = 0; target[1] = 1;}
            
      totSqrError += 0.5 * (pow(target[0] - output[0],2) + pow(target[1] - output[1],2));
        
      i++; total++;
    }
    
    
    // Update the number of epochs performed and check the stoppping criteria
		epochs++;
    if(epochs >= 1000 || (totSqrError/(float)total) <= 0.14) stop = true;

    
  } while(!stop); 

  // Return the results
  stats results;
  results.accuracy = (correct / (float)total) * 100;
  results.epochs = epochs;
  return results;   
}


float NeuralNet::classify(vector<rowdata*> & dataset)
{
  int correct = 0, total = 0;
  vector<rowdata*>::iterator i = dataset.begin();
  float * output;
  rowdata * example;
  cerr << "NeuralNet: Testing the neural network" << endl;
  
  while(i != dataset.end()) {
     // Forward propagate the input, which will propagate its output to the next layer and so on
    example = (*i);
    first->forwardProp(example->attributes);
    
    // Get the resulting output, check the classification
    output = last->getOutput();
    if(example->target == 0 && output[0] > output[1])  correct++;
    else if(example->target == 1 && output[1] > output[0]) correct++;
       
    total++; i++; 
  }
  

  // Return the results
  return (float)((correct / (float)total) * 100); 
}


