#include <Eigen>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
using Eigen::MatrixXd;
using namespace std;

/*  path variables  **

export EIGEN_PATH=/usr/local/include/eigen3/Eigen/ && export BOOST_PATH=/opt/homebrew/Cellar/boost/1.83.0/include/boost
g++ -I$EIGEN_PATH -I$BOOST_PATH /Users/aly/Desktop/Main/funni_code/machinelearning/neuralnet/nnMain.cpp -o nnMain && ./nnMain

**  compile statement  */

template<typename T>
T anyMax(T a, T b);

//generate random float
float randomFloat(float low, float high);

//sigmoid functions
float sigmoid(float f);
float sigDer(float f);

//input node struct
struct inputNode {
    //this will hold the input of floats
    vector<float> input;
    //input is 50D, so we initialize the vec for 50 vals all of value 0.0f
    inputNode() : input(50, 0.0f) {}
    //add input lmao
    void getInput(vector<float> in) {
        input = in;
    }
};

//hidden node struct
struct hiddenNode {
    //points back to inputnode so we know where to get input from
    inputNode* source;
    //points to output node so it can do calculations
    outputNode* end;

    //inputs
    vector<float> input;
    //inputweights
    vector<float> inputWeight;

    //output to pass onto next layer
    float output;

    //biases
    //jUnGKoOk iS mY BiAs brainrot
    float bias;

    //constructor
    hiddenNode(inputNode* s, outputNode* e) : source(s), inputWeight(initializeRandoms()), output(calculateOutput(), end(e)) {
        end->addHidden(this); //adds this to end node to link the 2
    } 

    vector<float> initializeRandoms() {
        //link to outputNode
        vector<float> temp;
        //set random weights
        for(int i = 0; i < 50; i++) { temp.push_back(randomFloat(-0.35f, 0.35f)); }
        //also sets bias
        bias = randomFloat(-0.35f, 0.35f);
        //return random weights
        return temp;
    }

    float calculateOutput() {
        //update input
        input = source->input;
        //set default output
        output = 0.0f;
        //add to output
        for(int i = 0; i < 50; i++) { output += inputWeight[i] * input[i]; }
        //add bias, return ReLU, add to output node through ReLU fxn
        output += bias;
        return ReLU(output);
    }

    //rectified linear unit activation protocol, note to self: biological neuron
    float ReLU(float weightedInputSum) {
        if(end) { end->addInput(anyMax(weightedInputSum, 0.0f)); }
        return anyMax(weightedInputSum, 0.0f);
    }

};

//binary output, therefore using a sigmoid function
struct outputNode {
    //outputs for calculations
    vector<float> hiddenOutputs;

    //pointers to hidden nodes for backpropagation
    vector<hiddenNode*> hiddenNodes;

    //bias, declared at random during initialization
    float bias;

    //constructor:
    outputNode() : bias(randomFloat(-0.35f, 0.35f)) {}

    //add output, accessible from hidden nodes
    void addHidden(hiddenNode* h) {
        hiddenNodes.push_back(h);
    }

    //add output float
    void addOutput(float f) {
        hiddenOutput.push_back(f);
    }

    //output + sigmoid function
    float returnFinalOutput() {
        float output = 0.0f;
        for(i = 0; i < hiddenOutputs.size(); i++) {
            output += hiddenOutputs[i];
        }
        output += bias;
        hiddenOutputs.clear();
        return sigmoid(output);
    }

};


class nodeNet {

    public:
        string name;
    

    float learningRate;


    nodeNet(string n, float r) : name(n), learningRate(r) {}



    void backPropagate(outputNode* end, float actual, float predicted) {
        //fixing input vals bc theyre trolling
        actual /= 4.0f;
        float loss = calculateLoss(actual, predicted);
        float gradient = calculateGradient(actual, predicted);
        //pass back through sigmoid function
        gradient *= sigDer(predicted);
        //apply to output node bias, learningRate is a class attribute 
        end->bias = end->bias - gradient*learningRate;
        //pass back to hidden nodes
        vector<hiddenNode*> allHidden = end->hiddenNodes;
        //apply to hidden node biases and weights
        //with relu, the chain rule is: *0 for <= 0, *1 for > 0; this applies to biases also
        //so change in bias, if relu does activate, will be the same as change in bias 
        //for the output node
        //however, for change in individual input weights, we need to multiply the changefactor
        //by the input's value. this is represented by index i in the loop below
        for(int i = 0; i < allHidden.size(); i++) {
            if(allHidden[i]->output > 0.0f) {
                reWeigh(allHidden[i], (gradient*learningRate));
            }
        }
    }

    void reWeigh(hiddenNode* hidden, float changeFactor) {
        for(int i = 0; i < hidden->inputWeight.size(); i++) {
            hidden->inputWeight[i] = hidden->inputWeight[i] - (changeFactor*hidden->inputp[i]);
            hidden->bias = hidden->bias - changeFactor;
        }
    }


    float calculateLoss(float actual, float predicted) {
        //fixing predicted for edgecases
        if(predicted <= 0.001f) predicted = 0.001f;
        if(predicted >= 0.999f) predicted = 0.999f;
        //binary cross entropy:
        return -(actual * log(predicted) + (1.0f - actual) * log(1.0f - predicted));
    }

    float calculateGradient(float actual, float predicted) {
        //fixing predicted for edgecases
        if(predicted <= 0.001f) predicted = 0.001f;
        if(predicted >= 0.999f) predicted = 0.999f;
        //calculate gradient
        return((predicted - actual)/(predicted * (1.0f - predicted)));
    }




/*
    Basically:
    1. compute loss
    2. compute loss gradient
    3. apply loss gradient to all weights and biases with: new w/b = old w/b - gradient*learning rate, learning rate = 0.007f;
    4. iterate
*/



};










//generate random float within bounds low and high
float randomFloat(float low, float high) {
    //seed
    random_device seed;
    mt19937 gen(seed());
    uniform_real_distribution<float> dist(low, high);
    return dist(gen);
}

//only used for the end node, therefore will be multiplied by 4 as sentiment rated 0 or 4 in the data
float sigmoid(float f) {
    return 1.0f/(1.0f + exp(-f));
}

float sigDer(float f) {
    return f*(1.0f - f);
}

//template swap
template<typename T>
T anyMax(T a, T b) {
    if(a > b) return a;
    return b;
}

int main() {
    MatrixXd mat(2,2);
    mat(0,0) = 1;
    mat(0,1) = 1;
    mat(1,0) = 1;
    mat(1,1) = 1;

    cout << mat << endl;

    float lmao = randomFloat(-0.35f, 0.35f);

    cout << lmao << endl;

    return 0;
}

