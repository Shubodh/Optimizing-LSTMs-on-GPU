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

void randMat(vector<vector<float> >& mat, int range)
{
	const int rows = mat.size(), cols = mat[0].size();
    int temp;
	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			temp = (rand() % range); 
			mat[i][j] = (temp - (range/2)); 
		}
	}
}


vector<vector<float> > matDim(int rows_a, int cols_a)
{
    vector<vector<float> > A(rows_a, vector<float>(cols_a));

	const int range = 100;
	randMat(A, range);

    return A;
}

void nextHiddenStateEfficient(vector<vector<float> >& input_t, vector<vector<float> >& h_tminus1, vector<vector<float> >& c_tminus1,int hiddenSize, int miniBatch)
{
        //dimension of ifog is now 2048.
        //vector<vector<float> > ifog_t_linear = sum_Wx_Rh_b(input_t, h_tminus1, hiddenSize, miniBatch);

        //W dimension here 2048 X 512 instead of 512 X 512
        vector<vector<float> > W = matDim(hiddenSize*4, hiddenSize);
        vector<vector<float> > R = matDim(hiddenSize*4, hiddenSize);
        // Wx dimensions will now be 2048 instead of 512
        vector<vector<float> > Wx = matDim(hiddenSize*4, miniBatch);
        vector<vector<float> > Rh = matDim(hiddenSize*4, miniBatch);

	omp_set_num_threads(2);

        int id;
	id = omp_get_thread_num();
	if(id == 0)
	{
		omp_set_num_threads(16);
		#pragma omp parallel for
			for(int i =0; i < W.size(); ++i){
        	            for (int j = 0; j < input_t[0].size(); ++j){
            			for (int k=0; k < input_t.size(); ++k){
                	            Wx[i][j] += W[i][k] * input_t[k][j];
        }
        }
        }
	}
	if(id == 1)
	{
		omp_set_num_threads(16);
		#pragma omp parallel for
			for(int i =0; i < R.size(); ++i){
        		    for (int j = 0; j < h_tminus1[0].size(); ++j){
            			for (int k=0; k < h_tminus1.size(); ++k){
                		    Rh[i][j] += R[i][k] * h_tminus1[k][j];
        }
        }
        }

	}
	

        
        vector<vector<float> > b = matDim(hiddenSize*4, miniBatch);

        //vector<vector<float> > sum1 = matSum(Wx,Rh);
	vector<vector<float> > sum1(Wx.size(), vector<float>(Rh[0].size())) ;
        for (int i =0; i < Wx.size(); ++i){
            for (int j = 0; j < Rh[0].size(); ++j){
                sum1[i][j] = Wx[i][j] + Rh[i][j];
        }
    }


        //vector<vector<float> > ifog_t_linear = matSum(sum1, b);
	vector<vector<float> > ifog_t_linear(sum1.size(), vector<float>(b[0].size())) ;
    for (int i =0; i < sum1.size(); ++i){
        for (int j = 0; j < b[0].size(); ++j){
                ifog_t_linear[i][j] = sum1[i][j] + b[i][j];
        }
    }


    //vector<vector<float> > ifog_t = matSigmaTanh(ifog_t_linear);
    vector<vector<float> > ifog_t(ifog_t_linear.size(), vector<float>(ifog_t_linear[0].size())) ;
    for (int i =0; i < 3 * (ifog_t_linear.size()/4); ++i){
        for (int j = 0; j < ifog_t_linear[0].size(); ++j){
                ifog_t[i][j] = 1 / (1 + exp(-ifog_t_linear[i][j]));
        }
    }
    
    for (int i = 3 * (ifog_t_linear.size()/4); i < ifog_t_linear.size(); ++i){
        for (int j = 0; j < ifog_t_linear[0].size(); ++j){
                ifog_t[i][j] = tanh(ifog_t_linear[i][j]);
        }
    }

    vector<vector<float> > i_t(hiddenSize, vector<float>(miniBatch)), f_t(hiddenSize, vector<float>(miniBatch)), o_t(hiddenSize, vector<float>(miniBatch)), g_t(hiddenSize, vector<float>(miniBatch));

    //extract_ifog(ifog_t, i_t, f_t, o_t, g_t);
    for (int i =0; i < (ifog_t.size()/4); ++i){
        for (int j = 0; j < ifog_t[0].size(); ++j){
                i_t[i][j] = ifog_t[i][j];
        }
    }
    
    for (int i =0; i < (ifog_t.size()/4); ++i){
        for (int j = 0; j < ifog_t[0].size(); ++j){
                f_t[i][j] = ifog_t[i+ (ifog_t.size()/4) ][j];
        }
    }
    
    for (int i =0; i < (ifog_t.size()/4); ++i){
        for (int j = 0; j < ifog_t[0].size(); ++j){
                o_t[i][j] = ifog_t[i + (2*(ifog_t.size()/4)) ][j];
        }
    }
    
    for (int i =0; i < (ifog_t.size()/4); ++i){
        for (int j = 0; j < ifog_t[0].size(); ++j){
                g_t[i][j] = ifog_t[i + (3*(ifog_t.size()/4)) ][j];
        }
    }

    

    //vector<vector<float> > temp_fOc = matMulElement(f_t, c_tminus1);
    vector<vector<float> > temp_fOc(f_t.size(), vector<float>(c_tminus1[0].size())) ;
    for (int i =0; i < f_t.size(); ++i){
        for (int j = 0; j < c_tminus1[0].size(); ++j){
                temp_fOc[i][j] = f_t[i][j] * c_tminus1[i][j];
        }
    }

    //vector<vector<float> > temp_iOg = matMulElement(i_t, g_t);
    vector<vector<float> > temp_iOg(i_t.size(), vector<float>(g_t[0].size())) ;
    for (int i =0; i < i_t.size(); ++i){
        for (int j = 0; j < g_t[0].size(); ++j){
                temp_iOg[i][j] = i_t[i][j] * g_t[i][j];
        }
    }

    //vector<vector<float> > c_t = matSum(temp_iOg, temp_fOc);
    vector<vector<float> > c_t(temp_iOg.size(), vector<float>(temp_fOc[0].size())) ;
    for (int i =0; i < temp_iOg.size(); ++i){
        for (int j = 0; j < temp_fOc[0].size(); ++j){
                c_t[i][j] = temp_iOg[i][j] + temp_fOc[i][j];
        }
    }

    //vector<vector<float> > tanh_c_t = matTanh(c_t);
    vector<vector<float> > tanh_c_t(c_t.size(), vector<float>(c_t[0].size())) ;
    for (int i =0; i < c_t.size(); ++i){
        for (int j = 0; j < c_t[0].size(); ++j){
                tanh_c_t[i][j] = tanh(c_t[i][j]);
        }
    }

    //vector<vector<float> > h_t = matMulElement(o_t, tanh_c_t);
    vector<vector<float> > h_t(o_t.size(), vector<float>(tanh_c_t[0].size())) ;
    for (int i =0; i < o_t.size(); ++i){
        for (int j = 0; j < tanh_c_t[0].size(); ++j){
                h_t[i][j] = o_t[i][j] * tanh_c_t[i][j];
        }
    }

    c_tminus1 = c_t;
    h_tminus1 = h_t;
}

double lstmNaiveEfficient(int hiddenSize, int miniBatch, int seqLength, int numLayers, int numRun) 
{
	struct timeval t1, t2;
	gettimeofday(&t1, 0);

    vector<vector<float> > input_t(hiddenSize, vector<float>(miniBatch)), h_tminus1(hiddenSize, vector<float>(miniBatch)), c_tminus1(hiddenSize, vector<float>(miniBatch));
    //initializing input vector
	const int range = 100;
	randMat(input_t, range);
    //initializing hidden and latent state
	randMat(h_tminus1, range);
	randMat(c_tminus1, range);
    for (int i = 0; i < seqLength; ++i){
        nextHiddenStateEfficient(input_t, h_tminus1, c_tminus1, hiddenSize, miniBatch);
    }
	
    gettimeofday(&t2, 0);

	double elapsedTime = (t2.tv_usec-t1.tv_usec);
	printf("Time for the run number NAIVE EFFICIENT %d :  %.8f ms \n\n", numRun, elapsedTime/1000000);

    return elapsedTime;
}

int main(int argc, char* argv[])
{
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
		printf("Usage: ./LSTM_opti_1 <seqLength> <numLayers> <hiddenSize> <miniBatch> <numRuns>\n");
		return 1;      
	}
	
    double naiveEffTime = 0.f;
	for (int run = 0; run < numRuns; run++) {
		naiveEffTime += lstmNaiveEfficient(hiddenSize, miniBatch, seqLength, numLayers, run);
	}

	printf("Average Runtime for LSTM NAIVE EFFICIENT is %.8f ms\n\n", naiveEffTime / (numRuns*1000000));

    return 0;
}