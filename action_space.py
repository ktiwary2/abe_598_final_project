import time
import gym
import gym_cap
import numpy as np
import constants
from numpy import linalg as la

class Action_space(object):
    """docstring for Action_space object."""

    """
    Initializes the Action space object so that actions can be generated.
    args:
        n_of_UAV: number of uavs
        n_of_UGV: number of ugvs
        isUnit_array: what each unit is: True if UAV, false o/w
    """
    def __init__(self, n_of_UAV, n_of_UGV, isUnit_array, env_instance=None ):
        self.n_of_UAV= n_of_UAV
        self.n_of_UGV= n_of_UGV
        assert len(isUnit_array) == self.n_of_UAV +  self.n_of_UGV
        self.unit_labels= isUnit_array
        self.uav_labels= []
        self.ugv_labels= []

        # in case we want action independent of the enviornment
        if env_instance is not None:
            self.env= env_instance

        for i in range (len(self.unit_labels)):
            if self.unit_labels[i]: #That index is UAV
                self.uav_labels.append(i)
            else:  #That index is UGV
                self.ugv_labels.append(i)

        self.uav_labels= np.array(self.uav_labels)
        self.ugv_labels= np.array(self.ugv_labels)
        print("uav_labels", self.uav_labels)
        print("ugv_labels", self.ugv_labels)

        self.env_map_done= False
        self.env_map= None
        self.next_action= []
        self.next_action.append([4,4,4,4,4,4])


    """
    Returns the next action the agents should take
    args:
        None
    returns:
        next action based on some heuristic
    """
    def get_action(self, obs=None, uav1location= None, uav2location= None):
        if(obs is not None):
            if (self.env_map_done):
                self._get_dual_input_action(obs)
            else:
                self._get_map_based_action(obs, uav1location, uav2location)

        return self.next_action[-1]

    """
    Sets the next action the agents will take. Appends it to the list.
    args:
        next_action
    """
    def set_action(self, next_action):
        self.next_action.append(next_action)

    """
    Find the next best action based only on the map.
    Calls set action to set the action.
    args:
        None
    returns:
        None
    """
    def _get_map_based_action(self, obs, uav1location= None, uav2location= None):
        #update Map
        if self.env_map is None:
            self.env_map= obs
        else:
            self._updateEnv_map(obs)
            print("updated Map")
        # Get an action based on curr obs
        uav_1_action, uav_2_action= self._best_Action_mapsearch(
                                        uav1location, uav2location)
        curr_action= 4* np.ones(self.n_of_UAV + self.n_of_UGV, dtype= np.int32)
        uav_1_index, uav_2_index= self.get_uav_index()
        curr_action[uav_1_index] = int(uav_1_action)
        curr_action[uav_2_index] = int(uav_2_action)
        print ('action', curr_action)
        self.set_action(curr_action)


    """
    Finds the next based action based on map AND observations of UGVs.
    args:
        None
    returns:
        None
    """
    def _get_dual_input_action(self, obs, method= None):
        print("now in get_dual_input_action")
        if(method is None or method == 'RRT'):
            self._get_RRT_based_actions()

        self.set_action(curr_action)
        assert 2==3
        pass

    """
    Updates the map as more observations come in.
    args:
        obs: observation
    returns:
        None
    """
    def _updateEnv_map(self, obs):
        self.thres_map= None
        for i in range (self.env_map.shape[0]):
            for j in range(self.env_map.shape[1]):
                self.env_map[i,j]= \
                            self._map_elem_operator(self.env_map[i,j],obs[i,j])

        if self.thres_map is not None:
            if( len(np.where(self.env_map == -1)[0]) < self.thres_map
                and len(np.where(self.env_map == constants.TEAM2_FLAG)[0] == 1) ):
                self.env_map_done = True
        else:
            if( len(np.where(self.env_map == constants.TEAM2_FLAG)[0] == 1) ):
                self.env_map_done = True
    """
    Evaluates the map and then returns the best action for UAV1 and UAV2.
    args:

    returns:
        uav1action: action for the first UAV
        uav2action: actions for the second UAV
    """
    def _best_Action_mapsearch(self, uav_1_location, uav_2_location):
        quadrant= np.argmin([
                        sum(self.env_map[0:,
                                         0:int(self.env_map.shape[1]/2)]),
                        sum(self.env_map[0:,
                                         int(self.env_map.shape[1]/2):])
                        ])
        print('quadrant', quadrant)
        # there are more unknown locations in quadrant 1 than in quadrant 2
        if (quadrant == 1):
            #implement this
            return (np.random.choice([0,1,2,3]),np.random.choice([0,1,2,3]))

        # there are more unknown locations in quadrant 2 than in quadrant 1
        else:
            quadrant= np.array([
                            sum(self.env_map[0:int(self.env_map.shape[0]),
                                             int(self.env_map.shape[1]/2):]),
                            sum(self.env_map[int(self.env_map.shape[0]/2):,
                                             int(self.env_map.shape[1]/2):])
                            ])
            center1= np.array([int((3*self.env_map.shape[1])/4),
                                int(self.env_map.shape[0]/4)]) # y,x

            ## actions 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay

            # which UAV is closer to the center (3y/4,x/4)
            # True if arg2 is closer than arg3
            if self._get_closest(center1, uav_1_location, uav_2_location) :
                # now check if closest UAV is to the right or the left of the center.
                if(uav_1_location[1] - center1[1]>0):
                    #move to the right and down
                   uav_1_action= np.random.choice([1,3]) # uav 1 to the left and down
                   uav_2_action= np.random.choice([1,2]) # uav 2 to the right and down
                else:
                    # move to the left and the down
                   uav_1_action= np.random.choice([1,2]) # uav 1 to the right and down
                   uav_2_action= np.random.choice([1,2]) # uav 2 to the right and down
            else:
                # now check if the closest UAV is to the right or the left of the center.
                if(uav_2_location[1] - center1[1]>0):
                    #move to the right and down
                   uav_2_action= np.random.choice([1,3]) # uav 1 to the left and down
                   uav_1_action= np.random.choice([1,2]) # uav 2 to the right and down
                else:
                    # move to the left and the down
                   uav_2_action= np.random.choice([1,2]) # uav 1 to the right and down
                   uav_1_action= np.random.choice([1,2]) # uav 2 to the right and down

        return (uav_1_action, uav_2_action)

    """
    Defines the actions of the the UGVs based on the information
    collected by the UAVs.
    """
    def _get_RRT_based_actions(self):
        ugv_locations= []
        for i in range (len(self.ugv_labels)):
            ugv_locations.append(self.env.team1[self.ugv_labels[i]].get_loc())
        ugv_locations= np.array(ugv_locations)
        print (ugv_locations)
        self._sample_next_states(ugv_locations)

    """
    Samples all possible paths from all ugv locations
    and find the optimal path by doing so 4 times.
    """
    def _sample_next_states(self, ugv_locations):

        pass

    """
    _map_elem_operator: Special "operator overload" to update old map with new info.
    args:
        a: a is current map value at that location
        b: the observation at that location
    Returns:
        the correct assignment given the current observation
    """
    def _map_elem_operator(self, a, b):
        if (constants.operator_dict.get((a,b), 'NONE') != 'NONE'):
            return constants.operator_dict[(a,b)]
        elif (constants.operator_dict.get(('xx',b), 'NONE') != 'NONE'):
            return constants.operator_dict[('xx',b)]
        elif (constants.operator_dict.get((a,'xx'), 'NONE') != 'NONE'):
            return constants.operator_dict[(a,'xx')]
        print('---------------------------------------')
        print('the following thing doesnt exist:', a,b)
        print('PROOF1:', constants.operator_dict.get((a,b), 'NONE'))
        print('PROOF2:', constants.operator_dict.get(('xx',b), 'NONE'))
        print('PROOF3:', constants.operator_dict.get((a,'xx'), 'NONE'))
        return constants.UNKNOWN

    """
    Miscelaneous methods:
    """
    def print_map(self):
        print ('MAP:', self.env_map)

    def _get_closest(self, center1, uav_1_location, uav_2_location):
        # if the euclidiean distance to uav is
        print ('center', center1)
        print('uav', uav_1_location)
        if(la.norm(center1-uav_1_location, 2) > la.norm(center1-uav_2_location, 2)):
            return True
        return False

    def get_uav_index(self):
        return (self.uav_labels[0], self.uav_labels[1])
