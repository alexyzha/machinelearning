/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   INCLUDES + NAMESPACE   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;
/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   COMPILE LINE   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

/*                                              g++ nnSecondDraft.cpp -std=c++11 -o nnSD && ./nnSD                                                */

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
struct inputNode;
struct hiddenNode;
struct outputNode;
struct line;
class nodeNet;
class tableStr;
/*––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   STRUCT ACT   ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
struct hiddenNode {

    float input; //changes every iteration
    float weight;
    float bias;
    float output;

    hiddenNode() {
        weight = randomFloat(-0.075f, 0.075f);
        bias = randomFloat(-0.075f, 0.075f);
        output = 0.0f;
    }

    float ReLU(float in) {
        if(in >= 0.0f) return in;
        return 0.0f;
    }

    void calculateOutput() {
        output += input*weight;
        output += bias;
        output = ReLU(input);
    }

    void getInput(float inp) {
        input = inp;
    }

};

struct outputNode {

    hiddenNode* hidden[50];
    vector<float> weights;
    float output;
    float bias;
    
    outputNode() : weights(50, randomFloat(-0.075f, 0.075f)), bias(randomFloat(-0.075f, 0.075f)) {}

    float calculateOutput() {
        float finalOutput = 0.0f;
        for(int i = 0; i < 50; i++) {
            hidden[i]->calculateOutput();
            finalOutput += hidden[i]->output * weights[i];
        }
        finalOutput += bias;
        finalOutput = sigmoid(finalOutput);
        output = finalOutput;
        return finalOutput;
    }

};

struct inputNode {

    vector<float> input;
    hiddenNode* hidden[50];

    inputNode() : input(50, 0.0f) {}

    void getInput(vector<float> in) {
        input = in;
        for(int i = 0; i < 50; i++) { hidden[i]->input = input[i]; }
    }

};

struct line {
    public:
    int actual;
    vector<float> embedded;
    line(int a, vector<float> e) : actual(a), embedded(e) {};
};

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––   CLASS ACTUAL   –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

class nodeNet {
public:
    float learningRate;
    inputNode* inputNeuron;
    outputNode* outputNeuron;
    nodeNet(float lr) : learningRate(lr) {
        inputNeuron = new inputNode();
        outputNeuron = new outputNode();
        for(int i = 0; i < 50; i++) {
            inputNeuron->hidden[i] = new hiddenNode();
            outputNeuron->hidden[i] = inputNeuron->hidden[i];
        }
    }

    void moveForward(line lstr) {
        inputNeuron->getInput(lstr.embedded);
        float predicted = outputNeuron->calculateOutput();
        backPropagate(static_cast<float>(lstr.actual), predicted);
    }

    void backPropagate(float actual, float predicted) {
        float loss = calculateLoss(actual, predicted);
        float gradient = calculateGradient(actual, predicted);
        //pass back through sigmoid function
        gradient *= sigDer(predicted);
        reWeigh(gradient*learningRate);
        //cout << setprecision(5) << loss << endl;
    }

    //auxillary functions
    void reWeigh(float changeFactor) {
        outputNeuron->bias -= changeFactor; //*outputNeuron->bias?
        for(int i = 0; i < 50; i++) {
            hiddenNode* hptr = outputNeuron->hidden[i];
            outputNeuron->weights[i] -= changeFactor*hptr->output; //IDK ABOUT THIS LINE
            if(hptr->output == 0.0f) continue;
            float tempCF = changeFactor*hptr->output;
            hptr->weight -= tempCF*hptr->input;
            hptr->bias -= tempCF*hptr->input;
        }
    }

    //loss function
    float calculateLoss(float actual, float predicted) {
        //fixing predicted for edgecases
        if(predicted <= 0.0001f) predicted = 0.0001f;
        if(predicted >= 0.9999f) predicted = 0.9999f;
        //binary cross entropy:
        return -(actual * log(predicted) + (1.0f - actual) * log(1.0f - predicted));
    }

    //loss gradient
    float calculateGradient(float actual, float predicted) {
        //fixing predicted for edgecases
        if(predicted <= 0.0001f) predicted = 0.0001f;
        if(predicted >= 0.9999f) predicted = 0.9999f;
        //calculate gradient
        return((predicted - actual)/(predicted * (1.0f - predicted)));
    }

};

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
    ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout.tie(NULL);

    //class object creation
    nodeNet neuralNet(0.007f);

    //train file:
    tableStr trainData;
    string filePath = "./nnData/fixedRandom/100k.csv";
    ifstream file(filePath);
    if(!file.is_open()) { cout << "bad\n"; return -1; }
    //add lines to trainData
    trainData.addLinesGlobal(file, 100000);
    file.close();
    //add more lines
    filePath = "./nnData/fixedRandom/randomBasic.csv";
    ifstream testfile(filePath);
    if(!testfile.is_open()) { cout << "bad\n"; return -1; }
    //add lines to testData
    trainData.addLinesGlobal(testfile, 20000);
    testfile.close();

    float avgdif = 0.0f;
    float correct = 0.0f;

    for(int i = 0; i < 120000; i++) {
        neuralNet.moveForward(trainData.alldata[i]);
        if(i > 100000) {
            avgdif += neuralNet.outputNeuron->output;
            if(abs(neuralNet.outputNeuron->output - static_cast<float>(trainData.alldata[i].actual)) < 0.5f) {
                cout << trainData.alldata[i].actual << endl;
                correct+=1.0f;
            }
        }
    }

    cout << endl << "Avg diff over runs 99000-100000: " << avgdif/20000.0f << endl << "percent correct: " << correct/200 << endl;


    exit(0);
    return 0;
}