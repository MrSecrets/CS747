import bandit
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
al = ["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
horizons = [100, 400, 1600, 6400, 25600, 102400]
fig, ax = plt.subplots()

ins = instances[2]
ins_value=np.loadtxt(ins)

regret1 = []
for hz in horizons:
	REG = 0
	for rs in range(0,50):
		reg = bandit.epsilon_greedy(ins_value,0.02,hz,rs)
		print(ins, al[0], rs, 0.02, hz, reg, sep=", ")
		REG += reg
	REG = REG/50
	regret1.append(REG)	
ax.plot(regret1, np.log(horizons))

regret2 = []
for hz in horizons:
	REG = 0
	for rs in range(0,50):
		reg = bandit.ucb(ins_value,0.02,hz,rs)
		print(ins, al[1], rs, 0.02, hz, reg, sep=", ")
		REG += reg
	REG = REG/50
	regret2.append(REG)	
ax.plot(regret2, np.log(horizons))

regret3 = []
for hz in horizons:
	REG = 0
	for rs in range(0,50):
		reg = bandit.kl_ucb(ins_value,0.02,hz,rs)
		print(ins, al[2], rs, 0.02, hz, reg, sep=", ")
		REG += reg
	REG = REG/50
	regret3.append(REG)	
ax.plot(regret3, np.log(horizons))

regret4 = []
for hz in horizons:
	REG = 0
	for rs in range(0,50):
		reg = bandit.thompson_sampling(ins_value,0.02,hz,rs)
		print(ins, al[3], rs, 0.02, hz, reg, sep=", ")
		REG += reg
	REG = REG/50
	regret4.append(REG)	
ax.plot(regret4, np.log(horizons))


ax.set_title('instance 3')
ax.legend(al)
ax.yaxis.set_label_text('log horizon')
ax.xaxis.set_label_text('regret')
plt.show()