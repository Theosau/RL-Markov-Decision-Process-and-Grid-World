import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Grid_World():
    """This class constructs the Grid World, transition probabililties and
       discount factor according to the CID.
       It provides th optimal policy and the actual value function states,
       based on the value iteration algorithm."""

    def __init__(self, cid):
        """This funciton sets-up the grid as well as the probabililtes,
           discount factor, and relevant matrices for the computation of
           the value iteration algorithm. It provides the position of all
           the unblocked states of the GridWorld of interest.
           input: an (8x1) numpy array with each CID digit."""

        # Grid World inputs initialisation
        self.cid = cid

        # Creating the dictionary of states and values
        self.states_names = ["s" + str(i+1) for i in range(11)]
        self.states_pos = np.array([[0, 0], [0, 1], [0, 2], [0, 3],
                                    [1, 0], [1, 1], [1, 3],
                                    [2, 1], [2, 2], [2, 3],
                                    [3, 2]])
        self.dic_states = dict(zip(self.states_names, self.states_pos))


        # Potential step directions
        self.directions = ["north", "south", "east", "west"]

        self.gamma = 0.2 + 0.5*0.1*cid[6]
        self.pba = 0.25 + 0.5*0.1*cid[5]
        self.pbo = (1-self.pba)/3
        self.v = np.zeros(len(self.states_names))
        self.policy = np.zeros(len(self.states_names))
        self.delta = 0
        self.diff = 10
        self.grid = np.ones((4 ,4))
        self.values = [0, 0, 0, 0]
        self.reward_state = self.dic_states[self.states_names[(cid[7]+1)%3]]


    def _new_states(self):
        """Figures out if the agent is in a terminal state and computes the
           potential new states it could take i.e all its neigbhours. Doing
           so, it ensures that the agent comes back to its current position
           if a transition would lead him outside of the Grid World boundary.

           output:
               - self.pot_new_state : a (4,) np.array of 4 (2,) arrays representing
                                   the potential new positions it could take.
               - self.terminal : bianary flag indicating if the state is terminal."""

        # flag to indicate if the current state is terminal
        self.terminal = False

        # checking if the current position is terminal
        if (np.all(self.current_state == self.reward_state) or \
            np.all(self.current_state == self.dic_states["s11"])):
            self.new_state = self.current_state
            self.terminal = True # terminal reached
            self.pot_new_state = []  #empty list of potential new states as it is static

        else:
            # transitions arrays: North, S, E, W
            self.translation = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
            # updating the potential positions
            self.pot_new_state = self.current_state + self.translation

            for i in range(len(self.pot_new_state)):
                # imposing the current position for out of bounds potential
                if not((self.pot_new_state[i].tolist() in self.states_pos.tolist())):
                    self.pot_new_state[i] = self.current_state

        return


    def _get_rewards(self):
        """Gathers the value for the rewards for each of the four potential
        new states in which the agent would  move. Recall that the if the
        agent is not in a terminal state, the reward is -100 getting to s11,
        +10 getting to the positive reward state (here s2), and -1 to any other
        state in the GridWorld. If it is in a terminal state, the reward is 0."""

        self.reward = np.zeros(4)

        for i in range(len(self.pot_new_state)):
            if self.terminal:
                self.reward[i] = 0  # 0 reward staying at termnial state
            else:
                # +10 if reaching the positive termnial state
                if np.all(self.pot_new_state[i] == self.reward_state):
                    self.reward[i] = 10

                # -100 reaching the negative termnial state
                elif np.all(self.pot_new_state[i] == self.dic_states['s11']):
                    self.reward[i] = -100

                # -1 for any other state
                else:
                    self.reward[i] = -1

        self.reward = np.reshape(self.reward, (4, 1))

        return


    def _get_value(self):
        """Copmutes the optimal value function at a given state and
           the policy which derived it. Performs the calculation of
           the Bellman Optimality Equation."""
        self.p = np.ones((4, 4))*self.pbo


        # input the diagonal terms of the probablility matrix to be the p
        # given by the CID, this is done to perfom the matrix calculation later
        # with the highest porbability given to the direction which matches the action

        for l in range (self.p.shape[0]):
            self.p[l, l] = self.pba

        # all 0s if the state is terminal
        if self.terminal:
            self.v[self.s] = 0
            self.policy[self.s] = np.nan
        else:
            self.values = np.zeros((4, 1))
            self.new_state = np.zeros(4)

            # get the index in the state position list
            # of each of the potential new states
            for n in range (self.new_state.shape[0]):
                self.new_state[n] = self.states_pos.tolist().index(self.pot_new_state[n].tolist())

            # gather the present value functions of each of
            # the potential states in a (4, 1) np array
            self.mini_values = np.reshape(np.array([self.v[int(self.new_state[0])],
                                                    self.v[int(self.new_state[1])],
                                                    self.v[int(self.new_state[2])],
                                                    self.v[int(self.new_state[3])]]),
                                          (4, 1))

            # value fucntions in matrix form
            self.values = np.matmul(self.p, (self.reward + self.gamma*self.mini_values))
            self.policy[self.s] = np.argmax(np.array(self.values))
            self.v[self.s] = np.max(self.values)

        return


    def _map_policy(self):
        """Maps the best policy given by numbers with letter names.
           Acts like a dictionary. """
        self.policies = ["north", "south", "east", "west"]
        self.policy_directions = []
        for i in range(self.policy.size):
            if np.isnan(self.policy[i]):
                # nan for the terminal states
                self.policy_directions.append("nan")
            else:
                self.policy_directions.append(self.policies[int(self.policy[i])])
        return

    def _value_grid_form(self):
        """Displays the grid as a (4x4) array to match the Grid World
           and clearly represent the value funciton at each state."""

        self.grid = np.nan*np.ones((4,4))
        for i in range(len(self.states_names)):
            self.x = self.states_pos[i].tolist()[0]
            self.y = self.states_pos[i].tolist()[1]
            self.grid[self.x, self.y] = self.v[i]

        return

    def _policy_grid_form(self):
        """Displays the grid as a (4x4) array to match the
           Grid World and clearly represent the policy"""

        l = [6, 8, 12, 13, 15]
        for j in range(len(l)):
            self.policy_directions.insert(l[j], 'nan')
        self.policy_directions = np.reshape(np.asarray(self.policy_directions),
                                            (4,4))

        return

    def _plot_grid(self):
        masked_grid = np.logical_not(np.isnan(self.grid))

        # for the plot, setting all the blocked states to -100
        cmap = colors.ListedColormap(['red','black','orange','yellow','green'])
        final_values_temp = self.grid.copy()
        final_values_temp[np.isnan(self.grid)]=-100
        maxi = self.grid[masked_grid].max()
        mini = self.grid[masked_grid].min()

        # positive and negative reward states
        final_values_temp[0, 1] = 500
        final_values_temp[3, 2] = -500
        bounds = [-500,-100,mini*1.1,0,maxi,500]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(final_values_temp, cmap=cmap, norm=norm, alpha=0.4)

        ax.set_xticks(np.arange(1, 0, 4))
        ax.set_yticks(np.arange(1, 0, 4))
        ax.grid(True)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if masked_grid[i, j]:
                    # add value on plot
                    plt.text(j-0.2, i, "{:.2f}".format(self.grid[i,j]))
                    if self.grid[i,j] != 0:
                        plt.text(j-0.4, i+0.2, "Go {}".format(self.policy_directions[i,j]))

        plt.title('Convergence of the Grid World')

        return

    def value_iteration(self):
        """Performs the value iteration algorithm up to a given
           convergence threshold delta.
        outputs:
            - final_values: np array if shape (11,) with the value
              function at each of the states
            - final_policy: np array if shape (11,) with the optimal
              policy at each of the states"""

        while self.diff > 0.01:
            self.v_old = self.v.copy()
            for self.s in range(len(self.states_names)):
                self.current_state = self.dic_states[self.states_names[self.s]]
                self._new_states()
                self._get_rewards()
                self._get_value()
            self.diff = max(self.delta, np.linalg.norm(self.v_old - self.v))
            self._map_policy()

        # formatting the converged solution
        self._policy_grid_form()
        self._value_grid_form()

        # plotting the solution
        self._plot_grid()

        return
