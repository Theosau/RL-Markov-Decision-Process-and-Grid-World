This repository contains the 1st Reinforcement Learning coursework from the Department of Computing, Imperial College London, Academic Year 2019-2020, delivered by Dr A. Aldo Faisal and Dr Edward Johns. The coursework was developed with their PhD students. 

# RL-Markov-Decision-Process-and-Grid-World

The goal of this project was to explore a Grid World setting and to solve for the optimal policy and value function at each states.

The Grid World of interest was a 4x4 grid, with state [0, 1] and [3, 2] as positive and negative reward states resepctively. The positions [1, 2], [2, 0], [3, 0], [3, 1], and [3, 3] were blocked, as shown in the image below.

<p align="center">
  <img src="images/grid_world.png" width=500>
</p>

Any move towards a state which is not terminal brings a reward of -1, whilst reaching the positive and negative termnial state provides rewards of 10 and -100 respectively. 
The agent could move in 4 different directions: North, South, East, West, and would stay in place (whilst still recieving the -1 reward) if its action led it out of the grid or to a blocked state. The agent greedily choses what direction it most move to based on the Bellman Optimality Equation whilst after having chosen a direction, the agents has a probability p=0.45 to move in this direction and a probability (1-p)/3 = 0.183 to move in a any of the three other directions.

The repository is as follows:

mdp_gridworld.py contains the annotated code which solves this problem. <br /> 
Notebook_to_run shows how to use the main code an provides the solution to this problem.
