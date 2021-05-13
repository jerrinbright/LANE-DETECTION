# Lane-detection-for-self-driving-cars

Autonomous lane detection for self-driving cars using Hough lines, Masking, Canny filters, and Gaussian filters
Primary libraries used in this project are OpenCv, NumPy, and matplotlib. 
This technology has a lot of real-life applications including self-driving cars to detect lanes accurately and traverse accordingly in real time.

The steps involved in this project are listed below:

Importing the required Libraries
Conversion of RGB to Grayscale
Gaussian Blur
Canny Filters
Masking
Combining Canny filters and masking images
Hough lines
Output image

For line by line explaining and detailing, you can view https://valueml.com/autonomous-lane-detection-for-self-driving-cars-in-python.


# LANE-DETECTION-USING-CNN

Lane detection for autonomous vehicles with the help of convolutional neural networks is done. We have previously worked on Lane detection using Canny edge and Hough lines. But they are not the best method/ technique that can be used for autonomous lane detection. For good performance of the canny detectors, we will have to mask the desired region of interest (ROI) which varies from one vehicle to another. Moreover, those filters and edge detectors tend to perform relatively poorly in high steep areas. Also, with real-life observations using monocular cameras shadows, glares and rapid movement of the vehicle tend to result in poor frame clarity resulting in inaccurate detection. Thus, the introduction of neural networks for lane detection tends to solve the above-mentioned issues and thereby increasing the robustness and accuracy of the system.

The pipeline of the architecture reflected in this project are:

Data-Preprocessing
Building Convolutional Neural Net
Training Model
Saving Model
Prediction in real-time

The article consists of 2 python files and thus two sections for the ease of understanding of the user. The first section/ python file will train the network and save the model. The second python file will implement the detection of lines in the input image/ video. We will use video in this article, as we will get video from monocular cameras in real-time for detection.

To know more about canny edge and hough lines, <a href="https://github.com/jerrie-bright/LANE-DETECTION"> Click here</a>

To understand the code in detail, refer my blog by <a href="valueml.com/lane-detection-using-neural-networks/"> clicking here</a>
