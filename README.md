# Lane detection 
## METHOD 1: Lane detection Using Hough Lines

Autonomous lane detection for self-driving cars using Hough lines, Masking, Canny filters, and Gaussian filters. Primary libraries used in this project are OpenCv, NumPy, and matplotlib. This technique has a lot of real-life applications including self-driving cars for detecting lanes accurately and traversing accordingly.<br>

The steps involved in this project are listed below:
<ul>
  <li>Importing the required Libraries</li>
  <li>Conversion of RGB to Grayscale</li>
  <li>Gaussian Blur</li>
  <li>Canny Filters</li>
  <li>Masking</li>
  <li>Combining Canny filters and masking images</li>
  <li>Hough lines</li>
  <li>Output image</li>
</ul>

The Method consists of 1 python script- ```canny_edge.ipynb``` which executes the proposed algorithm. 

## SHORTCOMINGS OF METHOD 1

Some assumptions/ shortcomings of Method 1 include: 
<ul>
  <li> Masking of the desired region of interest (ROI) which will vary from one vehicle to another. </li>
  <li> Filters and edge detectors tend to perform poorly in high steep areas.</li>
  <li> Shadows, glares and rapid movement of the vehicle results in poor frame clarity</li>
</ul>
  
## METHOD 2: Lane Detection using CNN

Lane detection for autonomous vehicles with the help of Convolutional Neural Networks (CNN) is experimented with in this section. The reasons/ shortcomings of Method 1 is discussed above and thus, CNNs where used to increase the robustness and reliability of the system.<br>

The pipeline of the architecture reflected in this Method is:

<ul>
  <li>Data-Preprocessing</li>
  <li>Building Convolutional Neural Net</li>
  <li>Training Model</li>
  <li>Saving Model</li>
  <li>Prediction in real-time</li>
</ul>
  
The Method consists of 3 python scripts- ```fully_conv_NN.py``` (training), ```draw_detected_lanes.py``` (testing) and ```test.ipynb``` for the ease of access. The first python script will train the network and save the model in h5 format (```full_CNN_model.h5```). The second python script will implement the detection of lines in the input image/ video and third python script will test the model.
