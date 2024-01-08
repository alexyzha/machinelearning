/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   INCLUDES + NAMESPACE   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
#include <Eigen>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
using Eigen::MatrixXd;
using namespace std;

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   COMPILE LINE   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
/*  path variables  **

export EIGEN_PATH=/usr/local/include/eigen3/Eigen/ && export BOOST_PATH=/opt/homebrew/Cellar/boost/1.83.0/include/boost
g++ -I$EIGEN_PATH -I$BOOST_PATH /Users/aly/Desktop/Main/funni_code/machinelearning/neuralnet/nnMain.cpp -std=c++11 -o nnMain && ./nnMain

**  compile statement  */
/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   AUXILLARY PROTOTYPES   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
//functions:
template<typename T>
T anyMax(T a, T b);
//generate random float
float randomFloat(float low, float high);
//sigmoid functions
float sigmoid(float f);
float sigDer(float f);

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   STRUCT PROTOTYPE   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
//declare structs
struct inputNode;
struct hiddenNode;
struct outputNode;
struct line;
//declare class
class nodeNet;
class tableStr;

/*––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   STRUCT ACT   ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
//input node struct
struct inputNode {
    public:
    //this will hold the input of floats
    vector<float> input;
    //input is 50D, so we initialize the vec for 50 vals all of value 0.0f
    inputNode() : input(50, 0.0f) {}
    //add input lmao
    void getInput(vector<float> in) {
        input = in;
    }
};

//binary output, therefore using a sigmoid function
struct outputNode {
    public:
    //outputs for calculations
    vector<float> hiddenOutputs;
    vector<float> weights;
    //pointers to hidden nodes for backpropagation
    vector<hiddenNode*> hiddenNodes;

    //bias, declared at random during initialization
    float bias;

    //constructor:
    outputNode() : bias(randomFloat(-0.015f, 0.05f)) {
        for(int i = 0; i < 10; i++) weights.push_back(randomFloat(-0.3f, 0.1f));
        for(int k = 0; k < 10; k++) weights.push_back(randomFloat(-0.05f, 0.35f));
    }

    //add output, accessible from hidden nodes
    void addHidden(hiddenNode* h) {
        hiddenNodes.push_back(h);
    }

    //add output float
    void addOutput(float f) {
        hiddenOutputs.push_back(f);
    }

    //output + sigmoid function
    float returnFinalOutput() {
        float output = 0.0f;
        for(int i = 0; i < hiddenOutputs.size(); i++) {
            output += hiddenOutputs[i]*weights[i];
        }
        output += bias;
        hiddenOutputs.clear();
        return sigmoid(output);
    }

};

//hidden node struct
struct hiddenNode {
    public:
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
    hiddenNode(inputNode* s, outputNode* e, float upperlimit, float lowerlimit) : source(s), end(e) {
        //initialize first randoms
        inputWeight = initializeRandoms(lowerlimit, upperlimit);
        output = calculateOutput();
        end->addHidden(this); //adds this to end node to link the 2
    } 

    vector<float> initializeRandoms(float upperlimit, float lowerlimit) {
        //link to outputNode
        vector<float> temp;
        //set random weights
        for(int i = 0; i < 50; i++) { temp.push_back(randomFloat(lowerlimit, upperlimit)); }
        //also sets bias
        bias = randomFloat(-0.035f, 0.035f);
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
        if(end) { end->addOutput(anyMax(weightedInputSum, 0.0f)); }
        return anyMax(weightedInputSum, 0.0f);
    }

};

//line
struct line {
    public:
    int actual;
    vector<float> embedded;
    line(int a, vector<float> e) : actual(a), embedded(e) {};
};

//table
class tableStr {
    public:
    vector<line> alldata;
    tableStr() : alldata() {}    

    //add line
    void addLine(int a, vector<float> e) {
        alldata.push_back(line(a, e));
    }

    //line reader
    void addLinesGlobal (ifstream& file, int totalLines) {
        for(int i = 0; i < totalLines; i++) {
            //temp line
            string tempLine;
            //to fill/upload
            int actual;
            vector<float> tempVec;
            //get entire line since its broken up
            getline(file, tempLine);
            //NA protection
            if(tempLine.size() < 5) continue;
            while(tempLine[tempLine.length()-2] != ']') {
                string tempTempLine;
                getline(file, tempTempLine);
                tempLine += " " + tempTempLine;
            }     
            //this is col(index)0
            actual = stoi(tempLine.substr(0));
            if(actual > 1) actual = 1;
            //in the data this is where all col(index)1 lines start
            int startingIndex = 4;
            //lambda
            auto convert = [&startingIndex](string line, int index) -> float {
                int end = index;
                while(line[end] != ' ' && line[end] != ']') end++;
                startingIndex = end;
                return stof(line.substr(index, end-index));
            };
            //converts string line into vector of 50
            while(startingIndex < tempLine.size()-2) {
                if(tempLine[startingIndex] == ']') break;
                if(tempLine[startingIndex] != ' ') {
                    float tempF;
                    tempF = convert(tempLine, startingIndex);
                    tempVec.push_back(tempF); }
                else startingIndex++;
            }
            addLine(actual, tempVec);
        }
    }
};

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   CLASS ACTUAL   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
//neural net class
class nodeNet {
    public:
    //global learning rate, declared at initialization by user
    float learningRate;
    inputNode* inputNeuron;
    outputNode* outputNeuron;
    hiddenNode* hiddenNeurons[20];
    //constructor
    nodeNet(float r) : learningRate(r) {
        //create pointer to input node first:
        inputNeuron = new inputNode();
        //create output node so we dont get clapped trying to set up hidden nodes
        outputNeuron = new outputNode();
        //create hidden nodes:
        for(int i = 0; i < 10; i++) {
            hiddenNeurons[i] = new hiddenNode(inputNeuron, outputNeuron, -0.025f, 0.005f);
            hiddenNeurons[i+10] = new hiddenNode(inputNeuron, outputNeuron, -0.005f, 0.025f);
        }
        //constructor done
    }

    void moveForward(inputNode*& in, outputNode*& end, line lstr) {
        in->input = lstr.embedded;
        end->hiddenOutputs.clear();
        //10 hidden nodes
        for(int i = 0; i < end->hiddenNodes.size(); i++) {
            end->hiddenNodes[i]->calculateOutput();
        }
        float predicted = end->returnFinalOutput();
        //cout << predicted << " --------- " << lstr.actual << endl;
        backPropagate(end, static_cast<float>(lstr.actual), predicted);
    }

    void backPropagate(outputNode*& end, float actual, float predicted) {
        //fixing input vals bc theyre trolling
        float loss = calculateLoss(actual, predicted);
        //cout << setprecision(9) << abs(loss) << endl;
        float gradient = calculateGradient(actual, predicted);
        //pass back through sigmoid function
        gradient *= sigDer(predicted);
        //apply to output node bias, learningRate is a class attribute 
        end->bias = end->bias - gradient*learningRate;
        for(int i = 0; i < end->hiddenNodes.size(); i++) end->weights[i] = end->weights[i] - gradient*learningRate;
        //pass back to hidden nodes
        vector<hiddenNode*> allHidden = end->hiddenNodes;
        //apply to hidden node biases and weights
        //with relu, the chain rule is: *0 for <= 0, *1 for > 0; this applies to biases also
        //so change in bias, if relu does activate, will be the same as change in bias 
        //for the output node
        //however, for change in individual input weights, we need to multiply the changefactor
        //by the input's value. this is represented by index i in the loop below
        for(int i = 0; i < allHidden.size(); i++) {
            //if hidden[index i]->output has an output >0, that means it passes relu and should
            //be updated. otherwise, we dont touch it at all and continue iterating
            if(allHidden[i]->output > 0.0f) {
                reWeigh(allHidden[i], (gradient*learningRate));
            }
        }
        //clear output node
        end->hiddenOutputs.clear();
    }

    //auxillary function
    void reWeigh(hiddenNode* hidden, float changeFactor) {
        for(int i = 0; i < hidden->inputWeight.size(); i++) {
            hidden->inputWeight[i] = hidden->inputWeight[i] - (changeFactor*hidden->input[i]);
            hidden->bias = hidden->bias - changeFactor;
        }
    }

    //loss function
    float calculateLoss(float actual, float predicted) {
        //fixing predicted for edgecases
        if(predicted <= 0.001f) predicted = 0.001f;
        if(predicted >= 0.999f) predicted = 0.999f;
        //binary cross entropy:
        return -(actual * log(predicted) + (1.0f - actual) * log(1.0f - predicted));
    }

    //loss gradient
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

/*––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   AUXILLARY FXNS   ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
//generate random float within bounds low and high
float randomFloat(float low, float high) {
    //seed
    static random_device seed;
    static mt19937 gen(seed());
    uniform_real_distribution<float> dist(low, high);
    return dist(gen);
}

//only used for the end node
float sigmoid(float f) {
    return 1.0f/(1.0f + exp(-f));
}

//sigmoid function derivative, used for backpropagation
float sigDer(float f) {
    return f*(1.0f - f);
}

//template max
template<typename T>
T anyMax(T a, T b) {
    if(a > b) return a;
    return b;
}

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   MAIN   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

int main() {
    //desync
    ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout.tie(NULL);
    //initializing neural net with learning rate of 0.007
    nodeNet neuralNet(0.007f);

    //make tablestruct with our dimensions
    tableStr trainData;

    //open filestream
    string filePath = "./nnData/fixedRandom/100k.csv";
    ifstream file(filePath);
    if(!file.is_open()) { cout << "bad\n"; return -1; }

    //add 2000 lines to trainData
    trainData.addLinesGlobal(file, 100000);

    float avgdif = 0.0f;
    float correct = 0.0f;


    for(int i = 0; i < 100000; i++) {
        neuralNet.moveForward(neuralNet.inputNeuron, neuralNet.outputNeuron, trainData.alldata[i]);
        if(i > 99000) {
            avgdif += neuralNet.outputNeuron->returnFinalOutput();
            if(abs(neuralNet.outputNeuron->returnFinalOutput() - static_cast<float>(trainData.alldata[i].actual)) < 0.5f) {
                correct+=1.0f;
            }
        }
    }

    cout << endl << "Avg diff over runs 99000-100000: " << avgdif/1000.0f << endl << "percent correct: " << correct/10 << endl;

    file.close();

    exit(0);
    return 0;
}
