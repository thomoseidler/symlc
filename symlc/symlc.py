"""
Module that provides symbolic computations for (limit cycle) oscillators
"""
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr #Parse string expressions of the input equations
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

#TODO Make a super class for ensembles.

##     https://stackoverflow.com/questions/49765174/solving-a-first-order-system-of-odes-using-sympy-expressions-and-scipy-solver

#TODO Linearization and Lyapunov stuff
#TODO Jacobian
#TODO Implement LieDerivative from notebook
#TODO Make a function that returns callables for slow_mf and LieDeriv so that
# plotting can be done in another module

class dynsys():
	"""
	Dynamical systems class.

	Dynsys class provides easy to initialize symbolic representations of
	dynamical systems.
	"""

	def __init__(self, equations, n_osc = 1):
		"""
		Constructor.

		Members:
			:List n: Degrees of each ODE in the system.
			:int n_eq: Number of differential equations (should be number of
			           independent variables).
			:int n_params: Number of parameters.
			:sympy.Symbol t: Symbolic variable for the time.

		Params:
			:param self: The object pointer.
			:List(string) equations: List of strings containing all differential
			                         equations.
			:int n_osc: (Optional) Number of oscillators (if an ensemble is
			                       involved). Defaults to 1.
		"""
		self.n_eq = len(equations)
		self.t = sp.symbols("t")
		self.equations = []
		for i in equations:
			self.equations.append(sp.Eq(parse_expr(i),0))
		self.n = self.degs()
		self.n_osc = n_osc

	def degs(self):
		"""
		Find the degrees of the differential equations.

		Find highest occurrence of xi. It is assumed that no system will have a degree higher than 9.
		Degree -1 means that no variable (x) occurs in the equation.

		Returns:
			:List(int) degrees:
		"""
		degs = []
		for idx, eq in enumerate(self.equations):
			s = eq.free_symbols	#Set of all symbols in the equation
			i = 10				#Like this, 9 will be the maximum degree of the system
			while sp.symbols('x{}_{}'.format(idx+1,i)) not in s and i>=0:
				i -= 1
			degs.append(i)
		return degs


	def variables(self):
		"""
		Find all variables (or derivatives) from system.

		Also return a set of equally named functions. They will be used
		internally for symbolic differentiation and stuff.

		Returns:
			:tuple containing:

			- List(string) variables
			- List(sympy.Functions) functions
		"""
		variables = []
		functions = []
		for idx, eq in enumerate(self.equations):
			s = eq.free_symbols	#Set of all symbols in the equation
			i = 0
			while i<=self.n[idx]:
				variables.append('x{}_{}'.format(idx+1,i))
				functions.append(sp.Function('x{}_{}'.format(idx+1,i))(self.t))
				i += 1
		return variables, functions


	def params(self):
		"""
		Find all parameters from system.

		Returns:
			:List(string) params:
		"""
		params = []
		for idx, eq in enumerate(self.equations):
			s = eq.free_symbols	#Set of all symbols in the equation
			i = 1
			while sp.symbols('p{}_{}'.format(idx+1,i)) in s:
				params.append('p{}_{}'.format(idx+1,i))
				i += 1
		return params

	def ode2dynsys(self):
		"""
		Make the ODEs to a larger system of first order ODEs.

		Returns:
			:List(sympy.Equations)
		"""
		var, fun = self.variables()	#They are already matching and can be used for substitution.
		dsys = []
		substitutes = []
		for i in zip(var,fun):
			substitutes.append(i)
		idx = 0
		for i in range(len(self.n)):
			if self.n[i]>0:
				for j in range(self.n[i]-1):
					dsys.append(sp.Eq(sp.diff(fun[idx],self.t),fun[idx+1]))
					idx += 1
			dsys.append(sp.Eq(sp.diff(fun[idx],self.t),
			                  sp.solve(self.equations[i],var[idx+1])[0]
							  ).subs(substitutes))
			idx += 2
		return dsys


	def integrable_dynsys(self, p):
		"""
		Return a list of first order equations for simulations.

		Returns:
			:List(sympy.Equations)
		"""
		var, fun = self.variables()	#They are already matching and can be used for substitution.
		dsys = []
		dyn_var = []
		idx = 0
		for i in range(len(self.n)):
			if self.n[i]>0:
				for j in range(self.n[i]-1):
					expr = sp.symbols(var[idx+1])
					dsys.append(expr)
					dyn_var.append(var[idx])
					idx += 1
			dyn_var.append(var[idx])
			dsys.append(sp.solve(self.equations[i].subs(p),var[idx+1])[0])
			idx += 2
		return dsys, dyn_var


	def make_integrable(self, p):
		dynsys, var = self.integrable_dynsys(p)	#TODO Somehow this must have a t-dependency for the integrator
		#var.append('t')
		#def eq(x, t):
		#	dydt = sp.lambdify(var, dynsys)#[sp.lambdify(var, i) for i in dynsys]
		#	return dydt
		eq = sp.lambdify(var, dynsys)#eq
		def f(x, t):
			return eq(*x)
		return f
#		dynsys, var = self.integrable_dynsys(p)
#		def eq(y, t):
#			s = zip(var,y)
#			#[expr.subs(s) for expr in dynsys]
#			dydt = []
#			for i in range(len(dynsys)):
#				dydt.append(dynsys[i].subs(s))
#				s = zip(var,y)			#TODO For some reason, s needs to be defined newly all the time, will be overwritten otherwise
#			return dydt
#		return eq


	def diff_dynsys(self,k):		# k is order of derivation
		"""
		Compute derivatives of the dynamical system and substitute with ODEs.

		Returns:
			:Two dimensional list(equations)
		"""
		dynsys = self.ode2dynsys()
		diff = []
		diff_out = [dynsys]
		substitutes = []
		l = len(dynsys)
		idx = 0
		for i in range(l):
			substitutes.append((dynsys[idx].lhs,dynsys[idx].rhs))
			idx += 1
		for i in range(1,k):
			idx = 0
			for j in range(l):
				diff.append(sp.Eq(sp.diff(dynsys[idx].lhs,self.t,i),
				            sp.diff(diff_out[i-1][idx].rhs,self.t
							).subs(substitutes)))
				idx += 1
			diff_out.append(diff)
			diff = []
		return diff_out


	def slow_mf(self):
		"""
		Compute slow manifold equation.

		Returns:
			:sympy.Equation
		"""
		dynsys = self.ode2dynsys()
		l = len(dynsys)
		dsys = self.diff_dynsys(l)
		M=sp.Matrix(l,l, lambda i,j: sp.simplify(dsys[j][i%l].rhs))
		slow_mf = sp.Eq( M.det(),0 )
		return slow_mf


	def plot_slow_mf(self,params,save=False,filename="slow_mf.png", show=True):
		"""
		Plot the slow manifold.

		:param params: Dict. Numerical values for parameters.
		:param save: Boolean. Save file? Defaults to False.
		:param filename: String. Name of the file if saved.

		Return sympy.plotting.plot.backend instance.
		"""
		var, fun = self.variables()	#They are already matching and can be used for substitution.
		substitutes = zip(fun,var)
		expr = self.slow_mf().subs(substitutes)
		expr2 = expr.subs(params).simplify()
		print(expr2)
		plot = sp.plotting.plot_implicit(expr2,show=False)
		backend = plot.backend(plot)
		backend.process_series()
		if save==True:
			backend.fig.savefig(filename, dpi=300)
		if show:
			backend.show()
		return backend


	def plot_slow_mf_and_simulation(self, params, save=False, filename="slow_mf.png", numerics={"time":[0,1,100], "initial_condition":[1,1]}):
		"""
		Plot the slow manifold and a simulation of the system for specified range and initial conditions.

		:param params: Dict. Numerical values for parameters.
		:param save: Boolean. Save file? Defaults to False.
		:param filename: String. Name of the file if saved.
	        :param numerics: dict. Contains timerange and initial conditions for simulation.

		"""

		#Check if numerics dict is correct
		if not numerics["time"] or not numerics["initial_condition"]:
			print("Please give valid numerics dictionary.")
		else:
			timespan = numerics["time"]
			ic = numerics["initial_condition"]

		#Prepare simulation
		f = self.make_integrable(params)
		t = np.linspace(*timespan)
		sol = spi.odeint(f, ic, t)

		backend = self.plot_slow_mf(params, save=False, show=False)
		backend.ax.scatter(sol[:, 0], sol[:, 1], c='r')
        
		if save==True:
			backend.fig.savefig(filename, dpi=300)
		backend.show()


	def plot_slow_mf_and_simulation_3d(self, params, save=False, filename="slow_mf.png", numerics={"time":[0,1,100], "initial_condition":[1,1,1]}, bbox=(-2.5, 2.5)):
		""" 		
		Plot the slow manifold of a three dimensional system and a simulation of the system for specified range and initial conditions.

		:param params: Dict. Numerical values for parameters.
		:param save: Boolean. Save file? Defaults to False.
		:param filename: String. Name of the file if saved.
	        :param numerics: dict. Contains timerange and initial conditions for simulation.
 
		"""
		from mpl_toolkits.mplot3d import Axes3D

		def plot_implicit(fn, bbox=(-2.5,2.5)):
			""" 
			create a plot of an implicit function 
			fn  ...implicit function (plot where fn==0)
			bbox ..the x,y,and z limits of plotted interval
			"""
			xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			A = np.linspace(xmin, xmax, 100) # resolution of the contour
			B = np.linspace(xmin, xmax, 15) # number of slices
			A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

			for z in B: # plot contours in the XY plane
				X,Y = A1,A2
				Z = fn(X,Y,z)
				cset = ax.contour(X, Y, Z+z, [z], zdir='z')
				# [z] defines the only level to plot for this contour for this value of z

			for y in B: # plot contours in the XZ plane
				X,Z = A1,A2
				Y = fn(X,y,Z)
				cset = ax.contour(X, Y+y, Z, [y], zdir='y')

			for x in B: # plot contours in the YZ plane
				Y,Z = A1,A2
				X = fn(x,Y,Z)
				cset = ax.contour(X+x, Y, Z, [x], zdir='x')

			# must set plot limits because the contour will likely extend
			# way beyond the displayed level.  Otherwise matplotlib extends the plot limits
			# to encompass all values in the contour.
			ax.set_zlim3d(zmin,zmax)
			ax.set_xlim3d(xmin,xmax)
			ax.set_ylim3d(ymin,ymax)

			#plt.show()
			return fig, ax

		var, fun = self.variables()
		substitutes = zip(fun,var)
		fn = self.slow_mf().lhs.subs(params)
		f = sp.lambdify(['x1_0','x2_0','x3_0'], fn.subs(substitutes)) #TODO geht das irgendwie systematischer?
		fig, ax = plot_implicit(f, bbox=bbox)

		f2 = self.make_integrable(params)
		t = np.linspace(*numerics["time"])
		y0 = numerics["initial_condition"]
		sol = spi.odeint(f2, y0, t)

		ax.scatter(sol.T[0],sol.T[1],sol.T[2])
		if save==True:
			fig.savefig(filename)
		plt.show()


# TODO
# 	def grid_sample_slow_mf(self, ranges, samples, threshold, substitutes):
# 		"""
# 		On a grid defined by ranges, check for all points (defined via
# 		sampling_interval),
# 		if the slow manifold equation has a result smalles than threshhold.
#
# 		:param ranges: List of tuples. Define range to sample in.
# 		:param samples: List of number of samples to take over each interval,
# 		                length has to match that of ranges.
# 		:param threshold: Float. Gives a threshold for marking points on slow
# 		                  manifold.
# 		"""
# 		axes = []
# 		for i, rng in enumerate(ranges):
# 			axes.append(np.linspace(*rng, samples[i]))
# 		grid = np.meshgrid(*axes)
# 		var, fun = self.variables()
# 		for i, v in enumerate(var):
# 			if v not in list(substitutes.keys()):
# 				substitutes[fun[i]] = v
# 			else:
#  				substitutes[fun[i]] = substitutes[v]
# 		slow_mf_eq = self.slow_mf().subs(substitutes)
# 		print(substitutes)
# #		print(slow_mf_eq)
# 		eq = sp.lambdify(slow_mf_eq.free_symbols, slow_mf_eq.lhs)
# 		return grid, eq(*grid)<threshold	#TODO Can slow_mf_eq be <0?
# 		#TODO Maybe it is better if a grid is returned which only keeps values where eq(*grid)<threshold==true
# 		#TODO Test!


def make_callable(expression):
	#TODO Works only for expressions that do not contain any functions of time!
	variables = sorted(expression.free_symbols, key = lambda symbol: symbol.name)
	print(variables)
	f = sp.lambdify(variables, expression)
	return f
