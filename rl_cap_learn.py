import time
import gym
import gym_cap
import numpy as np
from action_space import Action_space

## select the size of the map
env = gym.make("cap-v0") #start with a small one 20x20
#env = gym.make("cap-v1") #more challenging one 100x100

#observation = env.reset(0) # debug if you don't want random generated map

start_time = time.time()
done = False
step = 0
iteration = 0

# action_space= Action_space(2,2,[env.team1[0].air,
                                # env.team1[1].air,
                                # env.team1[2].air], env)

prev_obs= None

while iteration < 100:

    ## list of randomly generated actions can be generated like that
#    action = env.action_space.sample()  # choose random action

    ## to make only one unit doing action use that
    # action = [np.random.randint(0,5),4,4,4,4,4]
    # if iteration == 0:
        # action= action_space.get_action()

    # uav_1_index, uav_2_index= action_space.get_uav_index()
    # print('UAV index', uav_1_index, uav_2_index)
    # action= action_space.get_action(prev_obs, env.team1[uav_1_index].get_loc(), env.team1[uav_2_index].get_loc())
    ## actions 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay
#    print("Actions selected:",action)
    action= [np.random.choice([0,1,2,3]), np.random.choice([0,1,2,3]), np.random.choice([0,1,2,3]), np.random.choice([0,1,2,3])]
    observation, reward, done, info = env.step(action)  # feedback from environment
    print ("OBS", observation.shape)
    prev_obs= np.copy(observation)
    ## example of the data that you may use to solve the problem
#    print(observation) # print observation as a matrix (coded map)

    xa, ya = env.team1[0].get_loc() #location of unit 0 in list of friendly units
#    print("Unit 0 location:",xa,ya)
    print("Unit 0 is a UAV:", env.team1[0].air)
    print('info', info)
    print('reward', reward)
    print("Unit 0 on the map is given by:", prev_obs[ya][xa])
    # print(action_space.print_map())

    ## ANIMATION (if you want to see what is going on)
    env.render(mode="env") #obs, obs2,  or env
    ##watch the complete environment (you cannot use it to find a solution)
#    env.render(mode="env")
    time.sleep(2) #remove sleep if you don't want to watch the animation

    step += 1
    if step == 1000 or done:
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        env.reset()
        step = 0
        iteration += 1
