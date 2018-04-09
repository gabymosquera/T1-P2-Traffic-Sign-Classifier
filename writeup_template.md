#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/visual.png "Visualization"
[image2]: ./german-traffic-signs/1.jpg "Web Traffic Sign 1"
[image3]: ./german-traffic-signs/2.jpg "Web Traffic Sign 2"
[image4]: ./german-traffic-signs/3.jpg "Web Traffic Sign 3"
[image5]: ./german-traffic-signs/4.jpg "Web Traffic Sign 4"
[image6]: ./german-traffic-signs/5.jpg "Web Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gabymosquera/P2-Gaby/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a horizontal bar chart showing two data sets, the training data set and the validation data set, overlapped. The intent is to quickly visualize the quantity of each type of traffic sign within each set and to determine if there is a big difference proportionally speaking.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried different pre-processing techniques such as grayscale, normalization, and L2 regularization. I quickly realized, through trial and error, that the best result in my case was achieved when only normalization was used, and therefore that's the only pre-processing method I applied to all of my data sets. I did shuffle the data as well.

As a first step, I decided to normalize all of the images in all of the data sets available. I normalized the images from 0 to 1 to avoid any negative numbers and to aid in calculation. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| FLATTEN				|												|
| Fully connected		| Input = 400, Output = 120						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Input = 120, Output = 84						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Input = 84, Output = 43						|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a training rate of 0.001, AdamOptimizer as my optimizer method, 25 EPOCHS, and a batch size of 128. I used a LeNet architecture with two inputs, the image input an the keep prob input for the dropout layer and two hyperparameters: mu = 0 and sigma = .1.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To train the model I used a optimized training operation using the reduced mean of the cross entropy of the logits. The logits were calculated using the LeNet function architecture per batch. For both the training and the validation sets I used an evaluate function that uses an accuracy operation and loss operation to find the accuracy and loss in each set per EPOCH. 

My final model results were:
* Validation Accuracy = 0.956
* Validation Loss: 0.177
* Training Accuracy = 0.995
* Training Loss: 0.019
* Test Accuracy = 0.939
* Test Loss = 0.315


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture tried was the pure LeNet architecture. It was chosen due to previous use in the LeNet lab where we used it to classify black and white images of numbers (0-9).

* What were some problems with the initial architecture?

The initial architecture was overfitting and therefore I was obtaining a high training accuracy but low validation accuracy.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Because of overfitting the architecture was adjusted and two dropout layers where added. the first dropout layer was added after the first fully connected layer, the second dropout layer was added after the second fully connected layer. 

* Which parameters were tuned? How were they adjusted and why?

Dropout layers have only one parameter to tune, the keep probability paremeter that determines the percentage of the weights that are kept and the ones that are dropped. This is only used while training and a keep probability value of 1 needs to be used while evaluating. While training a used a keep probability value of 0.6 for both dropout layers. That value was determined after some trial and error runs with different values such as 0.8, 0.7, 0.5. 0.6 seemed to yield the best result and the least difference between the accuracies in the training and validation sets.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I think that two convolutional layers in my architecture worked well on this project since the images that we might be processing might have the traffic sign at the top, bottom, corners, or any other part of the image. Convoluting the images means the architectue scans the image from top to bottom that way gathering information from simple lines in the first layer to more complex shapes in the second one. Also finding the traffic sign regardless of where they are located in the overall image.

The use of dropouts was a huge improvement in my archittecture since the overfitting was yeilding a differenc of about .3 between my training accuracy and my validation accuracy. this helped close that gap to about .039.

If a well known architecture was chosen:
* What architecture was chosen? 
LeNet was chosen as the base of my architecture to train my neural networks

* Why did you believe it would be relevant to the traffic sign application?
The layers used in this architecture worked very well when classifying numbers from 0 to 9 in the LeNet lab. Working with traffic signs in this project was going to require some adjusting but it seemed like a good base to start with.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
All of he accuracies are above .93 which means that the program should be able to classify accurately about 93% or more of the times.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No passing   			| No passing									|
| Yield					| Yield											|
| 60 km/h	      		| 60 km/h   					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of the 5 images downloaded from the web

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is very sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop sign   									| 
| .0000006    			| No entry 										|
| .00000002 			| 60 km/h										|
| .000000004   			| Bicycles crossing				 				|
| .000000003		    | 30 km/h  										|


For the second image, the model is very sure that this is a no passing sign (probability of 0.999), and the image does contain a no passing sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| No passing   									| 
| .00006    			| Vehicles over 3.5 metric tons prohibited 		|
| .00000003 			| No passing for vehicles over 3.5 metric tons	|
| .000000004   			| No vehicles   				 				|
| .00000000006		    | Dangerous curve to the right					|

For the third image, the model is very sure that this is a yield sign (probability of 1), and the image does contain a yield sign. The top five soft max probabilities were 

| Probability         						|     Prediction	   					| 
|:-----------------------------------------:|:-------------------------------------:| 
| 1         								| Yield   								| 
| .0000000000000000000000000006				| Keep right 							|
| .0000000000000000000000000000003			| Priority Road							|
| .000000000000000000000000000000001 		| 60 km/h   					 		|
| .00000000000000000000000000000000000008	| No passing 							|

For the fourth image, the model is very sure that this is a speed limit 60 km/h sign (probability of 1), and the image does contain a speed limit 60 km/h sign. The top five soft max probabilities were 

| Probability         		|     Prediction	   					| 
|:-------------------------:|:-------------------------------------:| 
| 1         				| 60 km/h   							| 
| .000000000002				| 80 km/h 								|
| .000000000000000000009	| Dangerous curve to the right			|
| .0000000000000000000003	| End of all speed and passing limits	|
| .00000000000000000000007	| No passing 							|

For the fifth image, the model is very sure that this is a priority road sign (probability of 1), and the image does contain a priority road sign. The top five soft max probabilities were 

| Probability         		|     Prediction	   					| 
|:-------------------------:|:-------------------------------------:| 
| 1         				| Priority road   						| 
| .0000000000000000007		| Traffic signals 						|
| .00000000000000000001		| Stop									|
| .0000000000000000000001	| Right-of-way at the next intersection	|
| .000000000000000000007	| No entry 								|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


