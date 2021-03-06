# Optimizing-LSTMs-on-GPU
Implementation of the [paper](https://arxiv.org/abs/1604.01946) "Optimizing Performance of Recurrent Neural Networks on GPUs" in CUDA and OpenMP.

The naming of the CPP files has been done in accordance with this NVIDIA [blog](https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/#disqus_thread). For example, `LSTM_opti_4.cpp` corresponds to `Optimization 4: Pre-Transposing the Weight Matrix`. 
 
## Example implementation:

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

Comprehensive results for every optimization can be seen in our [report](https://iiitaphyd-my.sharepoint.com/:b:/g/personal/shubodh_sai_research_iiit_ac_in/EbY_UnRKatVJoN3HamJWKd0B74-6WeGzCzTdWIy0hxAHew?e=vo54Ve).
