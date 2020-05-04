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


vector<vector<float> > matDim(int rows_a, int cols_a){
    vector<vector<float> > A(rows_a, vector<float>(cols_a));

	const int range = 100;
	randMat(A, range);

    return A;
}

vector<vector<float> > transpose(vector<vector<float> >& A) {
    int i,j;
    vector<vector<float> > B(A[0].size(), vector<float>(A.size()))  ;
    for(i=0; i<A.size(); i++) {
        for(j=0; j<A[0].size(); j++) {
            B[j][i] = A[i][j];
        }
    }

    return B;
}

vector<vector<float> > sum_Wx_Rh_b(vector<vector<float> > input_t, vector<vector<float> > h_tminus1, int hiddenSize, int miniBatch, vector<vector<float> > W, vector<vector<float> > R){

    vector<vector<float> > Wx = matDim(hiddenSize, miniBatch);
    vector<vector<float> > Rh = matDim(hiddenSize, miniBatch);

    int size1 = W.size() * W[0].size() * input_t[0].size();
    int size2 = R.size() * R[0].size() * h_tminus1[0].size();
    vector<float> Wx_temp(size1);
    vector<float> Rh_temp(size2);

    int t = 0;
    

    omp_set_num_threads(2);

    int id;
    id = omp_get_thread_num();
    if(id == 0)
    {
    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < W.size(); i++)
    	{
	    for(int j = 0; j < W[0].size(); j++)
		{
		    for(int k = 0; k < input_t[0].size(); k++)
		    {
		        Wx_temp[t] = W[i][j] * input_t[i][k];
			t++;
		    }
		}
    	}

    
    	int k = 0;
    
    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < W[0].size(); i++)
    	{
		for(int j = 0; j < input_t.size(); j++)
		{
	    		Wx_temp[i] = Wx_temp[k + W[0].size()* input_t.size()];
	    		k++;
		} 
    	}

    
    	k = 0;

    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < Wx.size(); i++)
    	{
		for(int j = 0; j < Wx[0].size(); j++)
		{
	    		Wx[i][j] = Wx_temp[k];
	    		k++;
		} 
    	}
    }
    
    if(id == 1)
    {
    	t = 0;

    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < R.size(); i++)
    	{
	    for(int j = 0; j < R[0].size(); j++)
		{
		    for(int k = 0; k < h_tminus1[0].size(); k++)
		    {
		        Rh_temp[t] = R[i][j] * h_tminus1[i][k];
			t++;
		    }
		}
    	}

    
    	int k = 0;
    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < R[0].size(); i++)
    	{
		for(int j = 0; j < h_tminus1.size(); j++)
		{
	    		Rh_temp[i] = Rh_temp[k + R[0].size()* h_tminus1.size()];
	    		k++;
		} 
    	}

    	k = 0;

    	omp_set_num_threads(16);
    	#pragma omp parallel for
    	for(int i = 0; i < Rh.size(); i++)
    	{
		for(int j = 0; j < Rh[0].size(); j++)
		{
	    		Rh[i][j] = Rh_temp[k];
	    		k++;
		} 
    	}
    }
    
    //vector<vector<float> > Wx = matMul(W, input_t);
    //vector<vector<float> > Rh = matMul(R, h_tminus1);
    vector<vector<float> > b = matDim(hiddenSize, miniBatch);

    vector<vector<float> > sum1 = matSum(Wx,Rh);
    vector<vector<float> > sumAll = matSum(sum1, b);

    return sumAll;

}

void nextHiddenState(vector<vector<float> >& input_t, vector<vector<float> >& h_tminus1, vector<vector<float> >& c_tminus1,int hiddenSize, int miniBatch, vector<vector<float> > W_i, vector<vector<float> > W_f, vector<vector<float> > W_o, vector<vector<float> > W_g, vector<vector<float> > R_i, vector<vector<float> > R_f, vector<vector<float> > R_o, vector<vector<float> > R_g){
 
    vector<vector<float> > i_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch, W_i, R_i);
    vector<vector<float> > f_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch, W_f, R_f);
    vector<vector<float> > o_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch, W_o, R_o);
    vector<vector<float> > g_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch, W_g, R_g);

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
    vector<vector<float> > W_i = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > W_f = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > W_o = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > W_g = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > R_i = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > R_f = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > R_o = matDim(hiddenSize, hiddenSize);
    vector<vector<float> > R_g = matDim(hiddenSize, hiddenSize);

    vector<vector<float> > W_i_trans = transpose(W_i);
    vector<vector<float> > W_f_trans = transpose(W_f);
    vector<vector<float> > W_o_trans = transpose(W_o);
    vector<vector<float> > W_g_trans = transpose(W_g);
    vector<vector<float> > R_i_trans = transpose(R_i);
    vector<vector<float> > R_f_trans = transpose(R_f);
    vector<vector<float> > R_o_trans = transpose(R_o);
    vector<vector<float> > R_g_trans = transpose(R_g);

    for (i = 0; i < seqLength; ++i){
        nextHiddenState(input_t, h_tminus1, c_tminus1, hiddenSize, miniBatch, W_i_trans, W_f_trans, W_o_trans, W_g_trans, R_i_trans, R_f_trans, R_o_trans, R_g_trans);
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
