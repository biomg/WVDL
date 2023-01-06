# WVDL
Computational algorithms for identifying RNAs that bind to specific RBPs are urgently needed, and they can complement high-cost experimental methods.  <br>
In this study, we propose a weighted voting deep learning (WVDL) to predict RNA-protein binding sites.  <br>

# Dependency:
python 3.8.5 <br>
Pytorch 1.4.0 <br>

# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir.  <br>
It has 24 experiments of 21 RBPs, and we need train one model per experiment. <br>

# Supported GPUs
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs. <br> In addition, WVDL can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins.  <br>
          
It supports model training, testing. <br>

# Usage:
python main.py 

The main.py file contains positive and negative training and test sets

python motif.py

Draw the motif pictures from CNN parameters.

# NOTE
When you train WVDL on your own constructed benchmark dataset, if the training loss cannot converge, may other optimization methods, like SGD or RMSprop can be used to replace Adam in the code.  <br>

# Contact
Zhengsen Pan: zhengsenpan@foxmail.com <br>

# Updates:
1/6/2023 <br>
