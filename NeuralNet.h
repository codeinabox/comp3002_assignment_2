/*
Basic Neural Network library
*/

#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

typedef struct {
  float * attributes;
  float target;
} rowdata;

typedef struct {
  float accuracy;
  int epochs;
} stats;


/*
Neural Network Layer class
*/
class NNLayer {
public:
  NNLayer() {};
  NNLayer(int units, int inputs);
  void setPrev(NNLayer * p) {prev = p;}
  void setNext(NNLayer * n) {next = n;}
  float getWeight(int i, int o); 
  void setWeight(int i, int o, float w);
  void updateWeight(int i, int o, float wChange);
  void randomiseWeights();
  void forwardProp(float * inp);
  void backProp(float * error, const float & learnRate);
	float* getOutput() {return output;}
  int getNoUnits() {return no_units;}
    
private:
  int no_units, no_inputs;
  float * weights, * bias, * output, * delta, * input;
  NNLayer * next, * prev;
  float logSig(const float & x);
  float logSigDeriv(const float & x);
};

/*
Neural Network class
*/
class NeuralNet {
public:
  NeuralNet() {};
  NeuralNet(int no_inputs, int no_hidden, int no_output);
  void setLearnRate(float r) {learnRate = r; cerr << "NeuralNet: The learning rate is " << learnRate << endl; }
  void randomiseWeights();
  stats train(vector<rowdata*> & dataset);
  float classify(vector<rowdata*> & dataset);

private:
  NNLayer * first, * last;
  float learnRate;   
};


#endif

