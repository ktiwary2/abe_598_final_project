# Capture the Flag
## Course project

### Preparation
1) create virtual conda environment from yml file or manually
2) install OpenAI GYM and PyGame using pip if needed
3) install Cap-Env using python setup script (look at alias.txt file for example)
4) try it out by running cap_learn.py

### Task
1) design a control algorithm that generates control actions
2) format of actions is a vector of 6 integers from 0 to 4 that correspond to Up, Right, Down, Left, Stay
3) you may use only observations given by the environment
4) you may create the algorithm in cap_learn or in a separate class/function and call it from cap_learn.

### Rules
1) you control one of the teams (Blue)
2) UAVs (circles) cannot be killed, have large observation range and can travel over obstacles
3) UGVs (squares) can kill (on their home territory) or be killed (on enemy territory) if the meet an enemy UGV
4) flag can be captured by UGV only

### Goals
your goals are:
  * find the enemy flag
  * pass through enemy patrol
  * capture the enemy flag using ground units
  * survive

### Good practice
1) you may look at the Agent class if you are interested in the information beyond observations
2) you are not allowed to use any information but observations from environment and our own team env.team1

