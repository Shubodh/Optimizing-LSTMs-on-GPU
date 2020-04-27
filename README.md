# Optimizing-LSTMs-on-GPU
Implementation of the paper "Optimizing Performance of Recurrent Neural Networks on GPUs" in CUDA and OpenMP.

For now, it has two versions of naive implementation of LSTM, basically 2.1 section. The second version of naive implementation is more efficient, as described in the second para of 2.1 or Listing 1.

With default parameters, the naive will take a long time to run. Use lower dimensions first to run it faster, for example you can use the following:
```
./naive-LSTM 5 1 64 8 1
```

This will give the following results:
```

Time for the run number NAIVE 0 :  49.17900000 ms 

Average Runtime for LSTM NAIVE is 49.17900000 ms 

Time for the run number NAIVE EFFICIENT 0 :  28.01100000 ms 

Average Runtime for LSTM NAIVE EFFICIENT is 28.01100000 ms
```
Therefore, we can see the second version is more efficient.
