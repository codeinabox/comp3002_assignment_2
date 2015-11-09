/*
COMP3002 Artificial Intelligence : Assignment 2 Backpropagation Neural Network

Coded by Matt Attlee, SID 9816022

Last updated 27/05/2003 
*/

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "NeuralNet.h"


void crossValidate(vector<rowdata*> & dataset, NeuralNet & nnet)
{
  // Set the number of folds as 10 and determine the fold size
  // NOTE: If the dataset doesn't the divide evenly then the training set gets the remainder
  int folds = 10, fold_size = dataset.size() / folds;
  stats * trainStats = new stats[folds];
  float * testAccuracy = new float[folds];   
  
  cerr << "Main: Performing " << folds << " fold cross validation with a fold size of " << fold_size << endl;
  
  // Create a training and data test set
  // and keep track if we are in the test set fold 
  vector<rowdata*> train_data, test_data; 
  bool in_test = false;
  
  // Create a whole heap of iterators
  vector<rowdata*>::iterator current, end, test_start, test_end;
  end = dataset.end();
  test_start = dataset.begin();
  test_end = test_start + fold_size;
  
  // Loop through k folds, calculate the sums as we go
  float sumTrainA = 0, sumTestA = 0; int sumTrainE = 0; 
  for (int k = 1; k <= folds ; k++) {
    train_data.clear(); test_data.clear();  current = dataset.begin();
    
    cerr << "\n ### Fold no " << k << " ### " << endl;
    
		// Keep iterating till we get to the end
    while(current != end) {
		  // Check if we are in the test or training data
      if(current == test_start || (in_test && current != test_end)) {
        in_test = true; test_data.push_back(*current);
      }
      else if (current == test_end || (!in_test && current != test_start)) {
        in_test = false; train_data.push_back(*current);     
      }
      current++;
    }
    
    // Train network
    nnet.randomiseWeights();
    trainStats[k] = nnet.train(train_data);
    cout << "Training took " << trainStats[k].epochs << " epochs and was " << trainStats[k].accuracy << "% accurate\n"; 
    
    // Test network
    testAccuracy[k] = nnet.classify(test_data);
    cout << "Test data classification was " << testAccuracy[k] << "% accurate\n";
    
    // Point the test data pointers to the next fold
    test_start += fold_size; test_end += fold_size;
    
    // Update the sums
    sumTrainA += trainStats[k].accuracy;
    sumTrainE += trainStats[k].epochs; 
    sumTestA += testAccuracy[k];
  }
  
  // Calculate the mean
  float meanEpochs = (float) sumTrainE / (float) folds;
  float meanTrainA = sumTrainA / folds;
  float meanTestA = sumTestA / folds;
  
  // Calculate the standard deviation
  float sdEpochs = 0, sdTrainA = 0, sdTestA; 
  for (int k = 1; k <= folds ; k++) {
    sdEpochs += pow(meanEpochs - (float)trainStats[k].epochs, 2);
    sdTrainA += pow(meanTrainA - trainStats[k].accuracy, 2);
    sdTestA += pow(meanTestA - testAccuracy[k], 2);
  }
  
	sdEpochs = sqrt(sdEpochs / folds);
  sdTrainA = sqrt(sdTrainA / folds);
  sdTestA = sqrt(sdTestA / folds);
  
  cout << endl << " #### AVERAGE RESULTS ####\n";
  cout << "Epochs " << meanEpochs << " +/- " << sdEpochs << endl;
  cout << "Training set accuracy " << meanTrainA << "% +/- " << sdTrainA << endl;
  cout << "Test set accuracy " << meanTestA << "% +/- " << sdTestA << endl;
	
}


void preprocessData(vector<rowdata*> & dataset) 
{
  // Keep track of the min and max values for each attribute
  float * maxs = new float[8];  float * mins = new float[8];
  for(int j = 0; j < 8; j++) {
    mins[j] = NULL; maxs[j] = NULL;
  } 

  // Find the max and min values
  vector<rowdata*>::iterator i = dataset.begin();
  while(i != dataset.end()) {
    for(int j = 0; j < 8; j++) {
      // Check and update min value
      if ((*i)->attributes[j] < mins[j]) mins[j] = (*i)->attributes[j];
      // Check and update max value
      if ((*i)->attributes[j] > maxs[j]) maxs[j] = (*i)->attributes[j];
    }
    i++; 
  }

  // Using the max and min values normalise the data
  i = dataset.begin();
  while(i != dataset.end()) {
    for(int j = 0; j < 8; j++)
      (*i)->attributes[j] = ((*i)->attributes[j] - mins[j]) / (maxs[j] - mins[j]);
    i++; 
  }

  // Free up memory
  delete maxs; delete mins;
  
  cerr << "Main: Normalised the data between 0 and 1" << endl;
}


int main(int argc, char *argv[])
{
  // Check for sufficient arguments
  if(argc < 4) {
    std::cout << "example usage: " << argv[0] << " datafile hidden-units learning-rate" << endl;
    exit(-1);
  }
  
  // Get the learning rate and no of hidden units from the command line
  int no_hidden = atoi(argv[2]);
  float learning_rate = atof(argv[3]);
  
  // Turn datafile into stream and check the input file exists
  ifstream datafile(argv[1]);
  if (!datafile) {
    std::cout << "ERROR: the file " << argv[1] << " doesn't exist" << endl;
    exit(-1);
  }
  
  // Parse the datafile into a vector of row_data
  // Right now this is hard coded for the pima-diabetes data set
  char * temp = new char[20];
  vector<rowdata*> dataset;
  rowdata * r;
  
  while (datafile.good()) {
    r = new rowdata; r->attributes = new float[8];
    // Get the 8 attributes, store them in an array of floats
    for(int i = 0; i < 8; i++) {
      datafile.getline(temp,20,',');
      r->attributes[i] = atof(temp);
    }
    // Get the target value for that row
    datafile.getline(temp,20,'\n');
    r->target = atof(temp);
    dataset.push_back(r);  
  }

  dataset.pop_back(); // Hack to get rid of the extra blank row
  
  // Close the file streams and free up memory where possible 
  delete(temp);
  datafile.close();
  cerr << "Main: Read in " << dataset.size() << " rows of data successfully from " << argv[1] << endl;

  // Normalise data between values of 0 and 1
  preprocessData(dataset);

  // Create and initialize the neural network
  NeuralNet nnet = NeuralNet(8, no_hidden, 2);
  nnet.setLearnRate(learning_rate);
  nnet.randomiseWeights();
  
  // Perform a 10-Fold Cross Validation on the dataset with the neural network
  crossValidate(dataset, nnet);
  
  // Exit the program
  exit(0);
}

