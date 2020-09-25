import bandit
import numpy as np
import matplotlib.pyplot as plt

ins = np.loadtxt('../instances/i-3.txt')

regret = []
epsilons = []

epsilon = 0
while epsilon <=1:
	print(epsilon)
	REG = 0
	for seed in range(0,50):
		reg = bandit.epsilon_greedy(ins,ep = epsilon,hz=102400, rs=seed)
		REG += reg
	epsilon += 0.01
	regret.append(REG/50)
	epsilons.append(epsilon)
print("plot")
plt.plot(epsilons, regret)
plt.ylabel("regret")
plt.xlabel("epsilon")
plt.show()