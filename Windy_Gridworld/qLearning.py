import numpy as np
import argparse
import matplotlib.pyplot as plt
import windy_gridworld

max_steps=200
episodes = 200
episode_axis = np.arange(episodes)
# time_total= 0

def qLearning_choose(epsilon,action_space,state):
	#state os current location
    action=0
    global Q
    if np.random.uniform(0, 1) < epsilon: 
        action = np.random.choice(action_space) 
        # print(action)
    else: 
        action = np.argmax(Q[state[0],state[1]])
        # print(Q[state[0],state[1]])
        # print(1)
        # print(Q[state[0],state[1],:])
        # print(2)
        # print(Q[state[0],state[1],:]) 
    return action 

def update_qLearning(state, state2, reward, action, action2): 
    gamma = 0.95
    alpha = 0.85
    global Q
    Q_current = Q[state[0],state[1], action]
    Q_max = np.max(Q[state2[0],state2[1],:])
    Q[state[0],state[1], action] = Q_current+alpha*(reward+gamma*Q_max-Q_current) 

def qLearning(eps,n_actions,stoch_bool):
	# global time_total
	global time_axis
	global Q
	global episodes
	global n_action
	for rs in range(0,10):
		Q = np.zeros((7,10,n_actions))
		np.random.seed(rs)
		time_total = 0
		for episode in range(episodes):
			# print(episode)
			grid = windy_gridworld.windy_gridworld()
			state_1 = grid.current_loc
			action_1 = qLearning_choose(eps,n_actions,state_1)
			total_reward=0
			t =0
			# print(time_total)
			# while t<max_steps:
			while True:
				stoch = 0
				
				if n_actions == 4:
					if stoch_bool == 1:
						wind_stoch = np.random.rand(1)[0]
						if wind_stoch <=(1.0/3):
							stoch =1
						if wind_stoch >=(2.0/3):
							stoch = -1
					reward,state_2, done = grid.move_grid_4(action_1,stoch)
				
				if n_actions == 8:
					if stoch_bool == 1:
						wind_stoch = np.random.rand(1)[0]
						if wind_stoch <=(1.0/3):
							stoch =1
						if wind_stoch >=(2.0/3):
							stoch = -1				
					reward,state_2, done = grid.move_grid_8(action_1,stoch)

				total_reward+=reward
				action_2 = qLearning_choose(eps,n_actions,state_2)
				update_qLearning(state_1, state_2, reward, action_1, action_2)
				# print(state_1)
				# print(action_1)
				# print(Q[state_1[0],state_1[1],action_1])
				# if state_1[1]>=6: print(state_1, action_1)
				# if state_1 == [5,6]:
					# print("sdfghjmjhgfdsfgh")
				state_1 = state_2
				action_1 = action_2

				t+=1
				time_total += 1
				if done:
					print(episode)
					print(time_total)
					# print("dfghjklkjhgfdsdfghj,kjhgfdsdfghjhgfdsdfghjhgf")
					break
			time_axis[episode][rs]=time_total

plt.figure()

# Plotting 4 move with without stochastic wind
n_action = 4
Q = np.zeros((7,10,n_action)) #dimension of grid
time_axis = np.zeros((episodes,10))
qLearning(0.5,n_action,0)
time_axis=np.mean(time_axis,axis=1)
plt.plot(time_axis,episode_axis, label='4 move no stochastic wind')

# Plotting 4 move with stochastic wind
n_action = 4
Q = np.zeros((7,10,n_action)) #dimension of grid
time_axis = np.zeros((episodes,10))
qLearning(0.5,n_action,1)
time_axis=np.mean(time_axis,axis=1)
plt.plot(time_axis,episode_axis, label='4 move with stochastic wind')

# Plottinh kings move without stochastic wind
n_action = 8
Q = np.zeros((7,10,n_action)) #dimension of grid
time_axis = np.zeros((episodes,10))
qLearning(0.5,n_action,0)
time_axis=np.mean(time_axis,axis=1)
plt.plot(time_axis,episode_axis, label="king's move no stochastic wind")

# PLotting kings move with stochstic wind
n_action = 8
Q = np.zeros((7,10,n_action)) #dimension of grid
time_axis = np.zeros((episodes,10))
qLearning(0.5,n_action,1)
time_axis=np.mean(time_axis,axis=1)
plt.plot(time_axis,episode_axis, label="king's move with stochastic wind")

plt.xlabel("time step")
plt.ylabel("episodes")
plt.title("Q Learning")
plt.legend()
plt.show()