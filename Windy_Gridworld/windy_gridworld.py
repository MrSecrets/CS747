import numpy as np

class windy_gridworld():

	# def windy_gridworld(dims=[7,10] ,wind=0, start=[4,0],goal=[4,10]):
	def __init__(self):
		super(windy_gridworld,self).__init__()
		self.dims = (7,10)
		self.env = np.zeros(self.dims)		
		self.wind = np.array([0,0,0,1,1,1,2,2,1,0])
		# self.wind = np.array([0,0,0,0,0,0,0,0,0,0])
		self.start = [3,0]
		self.goal = [3,9]

		# check grid
		if len(self.wind)!=self.dims[1]:
			return invalid
		elif self.start[0]>self.dims[0] and self.start[0]>=0:
			return invalid
		elif self.start[1]>self.dims[1] and self.start[1]>=0:
			return invalid
		elif self.goal[0]>self.dims[0] and self.goal[0]>=0:
			return invalid
		elif self.goal[1]>self.dims[1] and self.goal[1]>=0:
			return invalid

		self.current_loc = self.start
		self.status = 0
		# print(grid, start, goal, wind)

	def move_grid_4(self,action,stoch):
	# north = 0, east = 1, south = 2, west = 3
		loc = self.current_loc
		if action==0:
			loc[0] -= 1
		elif action==1:
			loc[1]+=1
		elif action ==2:
			loc[0]+=1
		elif action==3:
			loc[1]-=1
	
		self.current_loc = loc

		if self.current_loc==self.goal:
			self.status = 1
			return 100, self.current_loc , self.status


		loc[0]-=self.wind[loc[0]] + stoch

		if loc[0]<0: loc[0] = 0 
		if loc[1]<0: loc[1] = 0 
		if loc[0]>=self.dims[0]: loc[0] = self.dims[0]-1
		if loc[1]>=self.dims[1]: loc[1] = self.dims[1]-1 

		self.current_loc=loc
		if self.current_loc==self.goal: 
			self.status = 1
			return 100, self.current_loc , self.status

		return -1, self.current_loc , self.status

	def move_grid_8(self,action,stoch):
	# north = 0, ne = 1, east = 2, se = 3 south=4, sw = 5, west=6, nw=7
		loc = self.current_loc
		if action==0:
			loc[0] -= 1
		elif action==1:
			loc[0] -= 1
			loc[1] += 1
		elif action ==2:
			loc[1] += 1
		elif action==3:
			loc[0] += 1
			loc[1] += 1
		elif action==4:
			loc[0] += 1
		elif action==5:
			loc[0] += 1
			loc[1] -= 1
		elif action==6:
			loc[1] -= 1
		elif action==7:
			loc[0] -= 1
			loc[1] -= 1

		self.current_loc = loc

		if self.current_loc==self.goal:
			self.status = 1
			return 100, self.current_loc , self.status

		loc[0]-=self.wind[loc[0]] + stoch

		if loc[0]<0: loc[0] = 0 
		if loc[1]<0: loc[1] = 0 
		if loc[0]>=self.dims[0]: loc[0] = self.dims[0]-1
		if loc[1]>=self.dims[1]: loc[1] = self.dims[1]-1 

		self.current_loc=loc
		if self.current_loc==self.goal: 
			self.status = 1
			return 100, self.current_loc , self.status

		
		return -1 , self.current_loc, self.status

windy_gridworld()
