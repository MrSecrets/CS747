import numpy as np
import argsparse
import matplotlib.pyplot as plt
import windy_gridworld

def sarsa_choose(Q,epsilon,action_space,state):
	#state os current location
    action=0
    if np.random.uniform(0, 1) < epsilon: 
        action = np.random.choice(action_space) 
    else: 
        action = np.argmax(Q[state, :]) 
    return action 

def update_sarsa(state, state2, reward, action, action2): 
    predict = Q[state, action] 
    target = reward + gamma * Q[state2, action2] 
    Q[state, action] = Q[state, action] + alpha * (target - predict) 

def sarsa(eps,alpha,n_actions,episodes,stoch_bool):
	Q = np.zeros((dims,n_actions))
	max_steps=100

	for episode in range(episodes):
		grid = windy_gridworld.windy_gridworld()
		action = sarsa_choose(Q,eps,n_actions,loc)
		total_reward=0
		
		while t<max_steps:
			stoch = 0
			
			if n_actions == 4:
				if stoch_bool = 1:
					wind_stoch = np.random.rand(1)[0]
					if wind_stoch <=(1.0/3):
						stoch =1:
					if wind_stoch >=(2.0/3):
						stoch = -1:
				reward,state_1, done = windy_gridworld.move_grid_4(action,stoch)
			
			if n_actions == 8:
				if stoch_bool = 1:
					wind_stoch = np.random.rand(1)[0]
					if wind_stoch <=(1.0/3):
						stoch =1:
					if wind_stoch >=(2.0/3):
						stoch = -1:				
				reward,state_1, done = windy_gridworld.move_grid_8(action,stoch)

			total_reward+=reward
			action_2 = sarsa_choose(Q,eps,n_actions,new_loc)
            update_sarsa(state_1, state_2, reward, action_1, action_2)
            state_1 = state_2
            action_1 = action_2
            
            t+=1
            if done:
            	break



