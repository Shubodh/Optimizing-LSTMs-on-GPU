#include <stdio.h>
#include <stdlib.h> 
#include <sys/time.h>
#include <time.h>
#include <math.h> 
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;

void randMat(vector<vector<float> >& mat, int range){
	const int rows = mat.size(), cols = mat[0].size();
    int temp;
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			temp = (rand() % range); 
			mat[i][j] = (temp - (range/2)); 
		}
	}
}

vector<vector<float> > matMul(vector<vector<float> >& mat1, vector<vector<float> >& mat2){
    vector<vector<float> > mat3(mat1.size(), vector<float>(mat2[0].size())) ;
    
    omp_set_num_threads(8);
    #pragma omp parallel for
    for (int i =0; i < mat1.size(); ++i){
        for (int j = 0; j < mat2[0].size(); ++j){
            for (int k=0; k < mat2.size(); ++k){
                mat3[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mat3;
}

vector<vector<float> > matMulElement(vector<vector<float> >& mat1, vector<vector<float> >& mat2){
    vector<vector<float> > mat3(mat1.size(), vector<float>(mat2[0].size())) ;
    for (int i =0; i < mat1.size(); ++i){
        for (int j = 0; j < mat2[0].size(); ++j){
                mat3[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    return mat3;
}


vector<vector<float> > matSum(vector<vector<float> >& mat1, vector<vector<float> >& mat2){
    vector<vector<float> > mat3(mat1.size(), vector<float>(mat2[0].size())) ;
    for (int i =0; i < mat1.size(); ++i){
        for (int j = 0; j < mat2[0].size(); ++j){
                mat3[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    return mat3;
}

vector<vector<float> > matSigma(vector<vector<float> >& mat1){
    vector<vector<float> > mat2(mat1.size(), vector<float>(mat1[0].size())) ;
    for (int i =0; i < mat1.size(); ++i){
        for (int j = 0; j < mat1[0].size(); ++j){
                mat2[i][j] = 1 / (1 + exp(-mat1[i][j]));
        }
    }
    return mat2;
}

vector<vector<float> > matTanh(vector<vector<float> >& mat1){
    vector<vector<float> > mat2(mat1.size(), vector<float>(mat1[0].size())) ;
    for (int i =0; i < mat1.size(); ++i){
        for (int j = 0; j < mat1[0].size(); ++j){
                mat2[i][j] = tanh(mat1[i][j]);
        }
    }
    return mat2;
}


vector<float> matVecMul(vector<vector<float> >& mat, vector<float>& vect){
    int n = mat.size();
    vector<float> out(mat.size()) ;
    for (int i =0; i < n; ++i){
        for (int k=0; k < mat[0].size(); ++k){
            out[i] += mat[i][k] * vect[k];
        }
    }
    return out;
}

vector<vector<float> > productMatDim(int rows_a, int cols_a, int rows_b, int cols_b){
    if (cols_a != rows_b){
		printf("Matrices cannot be multiplied, check dimensions. \n");
		exit(0);      
    }

    vector<vector<float> > A(rows_a, vector<float>(cols_a));
	vector<vector<float> > B(rows_b, vector<float>(cols_b));

	const int range = 100;
	randMat(A, range);
	randMat(B, range);

	vector<vector<float> > productAB = matMul(A, B);
    return productAB;
}

vector<vector<float> > matDim(int rows_a, int cols_a){
    vector<vector<float> > A(rows_a, vector<float>(cols_a));

	const int range = 100;
	randMat(A, range);

    return A;
}

vector<vector<float> > sum_Wx_Rh_b(vector<vector<float> > input_t, vector<vector<float> > h_tminus1, int hiddenSize, int miniBatch){

    vector<vector<float> > W = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > R = matDim(hiddenSize, hiddenSize);

    vector<vector<float> > Wx = matMul(W, input_t);
    vector<vector<float> > Rh = matMul(R, h_tminus1);
    vector<vector<float> > b = matDim(hiddenSize, miniBatch);

    vector<vector<float> > sum1 = matSum(Wx,Rh);
    vector<vector<float> > sumAll = matSum(sum1, b);

    return sumAll;

}

void nextHiddenState(vector<vector<float> >& input_t, vector<vector<float> >& h_tminus1, vector<vector<float> >& c_tminus1,int hiddenSize, int miniBatch){
 
    vector<vector<float> > i_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch);
    vector<vector<float> > f_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch);
    vector<vector<float> > o_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch);
    vector<vector<float> > g_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch);

    vector<vector<float> > i_t = matSigma(i_t_linear);
    vector<vector<float> > f_t = matSigma(f_t_linear);
    vector<vector<float> > o_t = matSigma(o_t_linear);
    vector<vector<float> > g_t = matTanh(g_t_linear);

    vector<vector<float> > temp_fOc = matMulElement(f_t, c_tminus1);
    vector<vector<float> > temp_iOg = matMulElement(i_t, g_t);
    vector<vector<float> > c_t = matSum(temp_iOg, temp_fOc);
    vector<vector<float> > tanh_c_t = matTanh(c_t);
    vector<vector<float> > h_t = matMulElement(o_t, tanh_c_t);
    
    c_tminus1 = c_t;
    h_tminus1 = h_t;
}

double lstmNaive(int hiddenSize, int miniBatch, int seqLength, int numLayers, int numRun) {
	struct timeval t1, t2;
	gettimeofday(&t1, 0);

    vector<vector<float> > input_t(hiddenSize, vector<float>(miniBatch)), h_tminus1(hiddenSize, vector<float>(miniBatch)), c_tminus1(hiddenSize, vector<float>(miniBatch));

    //initializing input vector
	const int range = 100;
	randMat(input_t, range);
   // initializing hidden and latent state
	randMat(h_tminus1, range);
	randMat(c_tminus1, range);
    int i;
    for (i = 0; i < seqLength; ++i){
        nextHiddenState(input_t, h_tminus1, c_tminus1, hiddenSize, miniBatch);
    }
	
    gettimeofday(&t2, 0);

	double time1 = abs(t2.tv_usec-t1.tv_usec);
	printf("Time for the run number %d :  %.8f us \n\n", numRun, time1/1000000);

    return time1;
}

int main(int argc, char* argv[]){
	int seqLength;
	int numLayers;
	int hiddenSize;
	int miniBatch; 
	int numRuns;

	if (argc == 6) {
		seqLength = atoi(argv[1]);
		numLayers =  atoi(argv[2]);
		hiddenSize =  atoi(argv[3]);
		miniBatch =  atoi(argv[4]);   
		numRuns =  atoi(argv[5]);   
	}
	else if (argc == 1) {
		printf("Running with default settings\n");
		seqLength = 100;
		numLayers = 4;
		hiddenSize = 512;
		miniBatch = 64;
        numRuns = 1;
	}
	else {
		printf("Usage: ./naiveLSTM <seqLength> <numLayers> <hiddenSize> <miniBatch> <numRuns>\n");
		return 1;      
	}


	double totalTime = 0.f;
	for (int run = 0; run < numRuns; run++) {
		totalTime += lstmNaive(hiddenSize, miniBatch, seqLength, numLayers, run);
	}

	printf("Average Runtime for LSTM naive is %.8fms\n", totalTime / (numRuns*1000000));

    return 0;
}
