import numpy as np
import argparse

def epsilon_greedy(ins, ep, hz):
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

def ucb(ins, ep, hz):
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
			ucb_at = emp_mean + np.sqrt(2*log(i)/pulls)
			arm_pick = np.argmax(ucb_at)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		

	REG = np.max(ins)*hz - total_reward
	return REG

def kl_divergence(p,q):
	return d = p*log(p/q) + (1-p)*log((1-p)/(1-q))

def kl_ucb(ins, ep, hz):
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
					increment = (1-q)/12
					KL = 0
					while q<1:
						KL_temp = kl_divergence(emp_mean[j],q)
						if KL_temp*pulls[j] < bound:
							if KL<KL_temp:
								KL = KL_temp
								q_ideal = q
						q = q + increment
					ucb_kl[j] = q

			arm_pick = np.argmax(ucb_kl)
			reward = np.random.binomial(n=1, p=ins[arm_pick])
			rewards[arm_pick] += reward
			pulls[arm_pick] += 1
			total_reward += reward		

	REG = np.max(ins)*hz - total_reward
	return REG

def thomson_sampling(ins, ep, hz):
	return

def thomson_hint(ins, ep, hz):
	return


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--instance", type=str, default='../instances/i-1.txt')
	parser.add_argument("--algorithm", type=str, default='epsilon-greedy')
	parser.add_argument("--randomSeed", type=int, default='1')
	parser.add_argument("--epsilon", type=float, default='0.5')
	parser.add_argument("--horizon", type=int, default='30')

	args = parser.parse_args()

	ins = np.loadtxt(args.instance)
	al = args.algorithm
	rs = args.randomSeed
	ep = args.epsilon
	hz = args.horizon
	print(ins)
	print(ins.size)


	if(al=='epsilon-greedy'):
		bandit = epsilon_greedy(ins, ep, hz)
		print(ins, al, rs, ep, hz, bandit, sep=",")
	
	elif(al=='ucb'):
		bandit = ucb(ins, ep, hz)
		print(ins, al, rs, ep, hz, REG, sep=",")
	
	elif(al=='kl-ucb'):
		bandit = kl_ucb(ins, ep, hz)
		print(ins, al, rs, ep, hz, REG, sep=",")
	
	elif(al=='thomson-sampling'):
		bandit = thomson_sampling(ins, ep, hz)
		print(ins, al, rs, ep, hz, REG, sep=",")
	
	elif(al=='thomson-sampling-with-hint'):
		bandit = thomson_hint(ins, ep, hz)
		print(ins, al, rs, ep, hz, REG, sep=",")
	
if __name__ == '__main__':
	main()