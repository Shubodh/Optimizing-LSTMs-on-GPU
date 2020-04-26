#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

using namespace std;



int main(){
    
    int s1 = 3; int s2 = 4;
    vector<vector<float>> m1 = matDim(s1, s2);
//    vector<vector<float>> m2 = matDim(s1, s2);
    vector<vector<float>> m3 = matTanh(m1);

    printMatrix(m1);
//    printMatrix(m2);
    printMatrix(m3);

    return 0;
}
