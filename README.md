# Lane detection 
## Lane detection Using Hough Lines

Autonomous lane detection for self-driving cars using Hough lines, Masking, Canny filters, and Gaussian filters. Primary libraries used in this project are OpenCv, NumPy, and matplotlib. This technology has a lot of real-life applications including self-driving cars to detect lanes accurately and traverse accordingly in real time.<br>

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

## Lane Detection using CNN

Lane detection for autonomous vehicles with the help of convolutional neural networks is done. We have previously worked on Lane detection using Canny edge and Hough lines. But they are not the best method/ technique that can be used for autonomous lane detection. For good performance of the canny detectors, we will have to mask the desired region of interest (ROI) which varies from one vehicle to another. Moreover, those filters and edge detectors tend to perform relatively poorly in high steep areas. <br>

Also, with real-life observations using monocular cameras shadows, glares and rapid movement of the vehicle tend to result in poor frame clarity resulting in inaccurate detection. Thus, the introduction of neural networks for lane detection tends to solve the above-mentioned issues and thereby increasing the robustness and accuracy of the system.<br>

The pipeline of the architecture reflected in this project are:

<ul>
  <li>Data-Preprocessing</li>
  <li>Building Convolutional Neural Net</li>
  <li>Training Model</li>
  <li>Saving Model</li>
  <li>Prediction in real-time</li>
</ul>
The article consists of 2 python files and thus two sections for the ease of understanding of the user. The first section/ python file will train the network and save the model. The second python file will implement the detection of lines in the input image/ video. We will use video in this article, as we will get video from monocular cameras in real-time for detection.

## My Blog

For line by line explaining and detailing, you can <a href="https://valueml.com/autonomous-lane-detection-for-self-driving-cars-in-python.">click here</a> to through my blog on lane detection. 
