# Biosignal Stars
BSPBSM 2022

This project is about a P300 BCI paradigm of navigating a Zumo robot in order to 
follow the lines. Project has different submodules, which can be summarized as 
paradigm, offline preprocessing and classification, online preprocessing and 
classification, and controlling the robot.

The offline folder contains all the steps in the offline pipeline. In the 
p300_paradigm.py script, the visual platform to collect the training data 
is provided. There are four arrows placed on the screen, describing the possible 
directions that the Zumo robot can move towards to: right, left, forward, and 
backward. At every trial, one of the arrows are selected as target and 
emphasized with the red color. This signals the user to fixate on the 
given arrow. Then the arrows are highlighted with yellow color randomly. 
While the arrow that the user fixates is highlighted, a P300 signal is produced 
in the user's brain. 

In the offline_preprocessing.py script, the raw data from numerous sessions 
are collected and concatenated. Later, the data is band-pass filtered, 
baseline corrected, common average referencing applied, and processed to 
smaller epochs. These epochs are later saved.


