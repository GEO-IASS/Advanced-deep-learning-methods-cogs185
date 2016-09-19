I have provided two functions that can handle data that can be fit into memory.
	fixed_point_model: the function for fixed-point model; 
	auto_context_model:	the function for the auto-context model;

For the POS and OCR dataset, the code has been specifically optimized since the dataset of POS is of very high dimension. The code and data of my experiments are located in: 
.\OCR; .\POS and .\hypertext;
You can produce the experimental results in the paper.

Example:
load trainData; 
load trainY;
[model, errRate, errRate2] = fixed_point_model(trainData, trainData, trainY, trainY, 3, 41, 2, '-s 5 -c 1', 0);
[model, errRate, errRate2] = auto_context_model(trainData, trainData, trainY, trainY, 3, 41, 2, '-s 5 -c 1', 0);


In this release, we have used the toolbox liblinear(http://www.csie.ntu.edu.tw/~cjlin/liblinear/), and the UGM toolbox by Mark Schmidt.
If you have any questions, please contact: Quannan Li quannan.li@gmail.com

Reference: Quannan Li, Jingdong Wang, David Wipf and Zhuowen Tu, "Fixed-Point Model for Structured Labeling", ICML 2013