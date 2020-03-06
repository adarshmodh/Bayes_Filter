1) We are given a robot operating in a 2D gridworld. Every cell in the gridworld is characterized by a color (0 or 1). The robot is equipped with a noisy odometer and a noisy color sensor. Given a stream of actions and corresponding observations, we have implemented a Bayes filter to keep track of the robot’s current position.

2) The starter.npz file inside data folder contains a binary color-map, a sequence of actions, a sequence of observations, and a sequence of the correct belief states.

3) The code bayes_filter_test.py returns the belief grid which is a grid representation of the belief, as well as a belief state which is a a 1x2 maximum likelihood estimate of the robot position
