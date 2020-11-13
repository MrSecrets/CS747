import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

def epsilon_greedy(ins, ep, hz, rs):
	np.random.seed(rs)
	n_arms = ins.size
	rewards = np.zeros(n_arms)
	pulls = np.zeros(n_arms)
	total_reward = 0

	for i in range(0, hz):
		move = np.random.uniform()

		if move<=ep:
			arm_pick = np.random.randint(0,n_arms)
		else:
			with np.errstate(divide='ignore', invalid='ignore'):
				emp_mean = rewards/pulls				
			emp_mean[np.isnan(emp_mean)] = 1
			arm_pick = np.argmax(emp_mean)

		reward = np.random.binomial(n=1, p=ins[arm_pick])
		rewards[arm_pick] += reward
		pulls[arm_pick] += 1
		total_reward += reward

	REG = np.max(ins)*hz - total_reward
	return REG

def ucb(ins,ep, hz,rs):
	np.random.seed(rs)
	n_arms = ins.size
	rewards = np.zeros(n_arms)
	pulls = np.zeros(n_arms)
	total_reward = 0

	for i in range(0,hz):
		if i<n_arms:
			arm_pick = i
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		
		else:
			with np.errstate(divide='ignore', invalid='ignore'):
				emp_mean = rewards/pulls				
			emp_mean[np.isnan(emp_mean)] = 1			
			ucb_at = emp_mean + np.sqrt(2*np.log(i)/pulls)
			arm_pick = np.argmax(ucb_at)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		

	REG = np.max(ins)*hz - total_reward
	return REG

def kl_divergence(p,q):
	return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def kl_ucb(ins,ep, hz,rs):
	np.random.seed(rs)
	n_arms = ins.size
	rewards = np.zeros(n_arms)
	pulls = np.zeros(n_arms)
	ucb_kl = np.zeros(n_arms)
	total_reward = 0

	for i in range(0,hz):
		if i<n_arms:
			arm_pick = i
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		
		else:
			with np.errstate(divide='ignore', invalid='ignore'):
				emp_mean = rewards/pulls				
			emp_mean[np.isnan(emp_mean)] = 1
			bound = np.log(i) + 3*np.log(np.log(i))
			for j in range(0, n_arms):
				if emp_mean[j] == 1:
					ucb_kl[j] =1
				else:
					q = emp_mean[j]
					q_ideal = q
					increment = (0.99-q)/12
					KL = 0
					while q<0.99:
						KL_temp = kl_divergence(emp_mean[j],q)
						if KL_temp*pulls[j] < bound:
							q_ideal = q
						q = q + increment
					ucb_kl[j] = q_ideal

			arm_pick = np.argmax(ucb_kl)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		

	REG = np.max(ins)*hz - total_reward
	return REG

def thompson_sampling(ins,ep, hz,rs):
	np.random.seed(rs)
	n_arms = ins.size
	rewards = np.zeros(n_arms)
	s_at = np.zeros(n_arms)
	f_at = np.zeros(n_arms)
	x_at = np.zeros(n_arms)
	total_reward = 0

	for i in range(0, hz):
		for j in range(0,n_arms):
			x_at[j] = np.random.beta(a=s_at[j]+1, b=f_at[j]+1)
		arm_pick = np.argmax(x_at)
		reward = np.random.binomial(n=1, p=ins[arm_pick])
		if reward:
			s_at[arm_pick] += 1
		else:
			f_at[arm_pick] += 1
		rewards[arm_pick] += reward
		total_reward += reward

	REG = np.max(ins)*hz - total_reward
	return REG

def thompson_hint(ins,ep, hz, rs):
	np.random.seed(rs)
	n_arms = ins.size
	rewards = np.zeros(n_arms)
	s_at = np.zeros(n_arms)
	f_at = np.zeros(n_arms)
	x_at = np.zeros(n_arms)
	true_mean_ls = ins
	total_reward = 0

	for i in range(0, hz):
		move = np.random.uniform()

		if move<=ep:
			arm_pick = np.argmax(true_mean_ls)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			if reward:
				s_at[arm_pick] += 1
			else:
				f_at[arm_pick] += 1
		else:
			for j in range(0,n_arms):
				x_at[j] = np.random.beta(a=s_at[j]+1, b=f_at[j]+1)
			arm_pick = np.argmax(x_at)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			if reward:
				s_at[arm_pick] += 1
			else:
				f_at[arm_pick] += 1
		
		rewards[arm_pick] += reward
		total_reward += reward

	REG = np.max(ins)*hz - total_reward
	return REG

	return 0


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--instance", type=str, default='../instances/i-1.txt')
	parser.add_argument("--algorithm", type=str, default='thompson-sampling', help='algorithm')
	parser.add_argument("--randomSeed", type=int, default='100')
	parser.add_argument("--epsilon", type=float, default='0.5')
	parser.add_argument("--horizon", type=int, default='30')
	# parser.add_argument("--true_mean", type=float, default='[0,0]')

	args = parser.parse_args()

	ins = np.loadtxt(args.instance)
	al = args.algorithm
	rs = args.randomSeed
	ep = args.epsilon
	hz = args.horizon
	# tm = args.true_mean

	np.random.seed(rs)


	if(al=='epsilon-greedy'):
		bandit = epsilon_greedy(ins, ep, hz, rs)
		print(args.instance, al, rs, ep, hz, bandit, sep=", ")
	
	elif(al=='ucb'):
		bandit = ucb(ins, ep, hz, rs)
		print(args.instance, al, rs, ep, hz, bandit, sep=", ")
	
	elif(al=='kl-ucb'):
		bandit = kl_ucb(ins, ep, hz, rs)
		print(args.instance, al, rs, ep, hz, bandit, sep=", ")
	
	elif(al=='thompson-sampling'):
		bandit = thompson_sampling(ins, ep, hz, rs)
		print(args.instance, al, rs, ep, hz, bandit, sep=", ")
	
	elif(al=='thompson-sampling-with-hint'):
		bandit = thompson_hint(ins, ep, hz, rs)
		print(args.instance, al, rs, ep, hz, bandit, sep=", ")
	
if __name__ == '__main__':
	main()
