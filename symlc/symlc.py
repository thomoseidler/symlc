"""
Module that provides symbolic computations for (limit cycle) oscillators
"""
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
)
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

"""Dynamical systems class.

	Dynsys class provides easy to initialize symbolic representations of
	dynamical systems.
"""


class dynsys:
    def __init__(self, equations, n_osc=1):
        """
		Constructor.

		Members
		-------
		n: list
			Degrees of each ODE in the system.
		n_eq: int
			Number of differential equations (should be number of
			           independent variables).
		n_params: int
			Number of parameters.
		t: sympy.Symbol
			Symbolic variable for the time.

		Parameters
		----------
		equations: List(string)
			List of strings containing all differential
			                         equations.
		n_osc: int
			(Not fully implemented) Number of oscillators (if an ensemble is involved). Defaults to 1.
		"""
        self.n_eq = len(equations)
        self.t = sp.symbols("t")
        self.equations = []
        for i in equations:
            self.equations.append(sp.Eq(parse_expr(i), 0))
        self.n = self.degs()
        self.n_osc = n_osc

    def degs(self):
        """Find the degrees of the differential equations.
		
		Find highest occurrence of xi. It is assumed that no system will have a degree higher than 9.
		Degree -1 means that no variable (x) occurs in the equation.

		Returns
		-------
		degrees: list(int)

		"""
        degs = []
        for idx, eq in enumerate(self.equations):
            s = eq.free_symbols  # Set of all symbols in the equation
            i = 10  # Like this, 9 will be the maximum degree of the system
            while sp.symbols("x{}_{}".format(idx + 1, i)) not in s and i >= 0:
                i -= 1
            degs.append(i)
        return degs

    def variables(self):
        """Find all variables (or derivatives) from system.
		
		Also return a set of equally named functions. They will be used
		internally for symbolic differentiation and stuff.

		Returns
		-------

		    tuple containing:
		    - List(string) variables
		    - List(sympy.Functions) functions

		"""
        variables = []
        functions = []
        for idx, eq in enumerate(self.equations):
            s = eq.free_symbols  # Set of all symbols in the equation
            i = 0
            while i <= self.n[idx]:
                variables.append("x{}_{}".format(idx + 1, i))
                functions.append(sp.Function("x{}_{}".format(idx + 1, i))(self.t))
                i += 1
        return variables, functions

    def params(self):
        """Find all parameters from system.

		Returns
		-------
		    params: list(string)

		"""
        params = []
        for idx, eq in enumerate(self.equations):
            s = eq.free_symbols  # Set of all symbols in the equation
            i = 1
            while sp.symbols("p{}_{}".format(idx + 1, i)) in s:
                params.append("p{}_{}".format(idx + 1, i))
                i += 1
        return params

    def ode2dynsys(self):
        """Make the ODEs to a larger system of first order ODEs.

		Returns
		-------
		    List(sympy.Equations)

		"""
        var, fun = self.variables()
        dsys = []
        substitutes = []
        for i in zip(var, fun):
            substitutes.append(i)
        idx = 0
        for i in range(len(self.n)):
            if self.n[i] > 0:
                for j in range(self.n[i] - 1):
                    dsys.append(sp.Eq(sp.diff(fun[idx], self.t), fun[idx + 1]))
                    idx += 1
            dsys.append(
                sp.Eq(
                    sp.diff(fun[idx], self.t),
                    sp.solve(self.equations[i], var[idx + 1])[0],
                ).subs(substitutes)
            )
            idx += 2
        return dsys

    def integrable_dynsys(self, p):
        """Return a list of first order equations for simulations.

		Parameters
		----------
		p : dict
			(key, val) pairs for the parameters
		    

		Returns
		-------
	    list(sympy.Equations), sympy.Variables

		"""
        var, fun = self.variables()
        dsys = []
        dyn_var = []
        idx = 0
        for i in range(len(self.n)):
            if self.n[i] > 0:
                for j in range(self.n[i] - 1):
                    expr = sp.symbols(var[idx + 1])
                    dsys.append(expr)
                    dyn_var.append(var[idx])
                    idx += 1
            dyn_var.append(var[idx])
            dsys.append(sp.solve(self.equations[i].subs(p), var[idx + 1])[0])
            idx += 2
        return dsys, dyn_var

    def make_integrable(self, p):
        """
		Make equations usable with `scipy.odeint`.

		Parameters
		----------
		p : dict
			(key, val) pairs with  parameter values.
		    

		Returns
		-------
		f: function
		"""
        dynsys, var = self.integrable_dynsys(p)
        eq = sp.lambdify(var, dynsys)

        def f(x, t):
            return eq(*x)

        return f

    def diff_dynsys(self, k):
        """Compute derivatives of the dynamical system and substitute with ODEs.

		Parameters
		----------
		k : int
			Order of the derivative
		    

		Returns
		-------
		two dimensional list(equations)

		"""
        dynsys = self.ode2dynsys()
        diff = []
        diff_out = [dynsys]
        substitutes = []
        l = len(dynsys)
        idx = 0
        for i in range(l):
            substitutes.append((dynsys[idx].lhs, dynsys[idx].rhs))
            idx += 1
        for i in range(1, k):
            idx = 0
            for j in range(l):
                diff.append(
                    sp.Eq(
                        sp.diff(dynsys[idx].lhs, self.t, i),
                        sp.diff(diff_out[i - 1][idx].rhs, self.t).subs(substitutes),
                    )
                )
                idx += 1
            diff_out.append(diff)
            diff = []
        return diff_out

    def slow_mf(self):
        """Compute slow manifold equation.

		Returns
		-------
		
		    sympy.Equation

		"""
        dynsys = self.ode2dynsys()
        l = len(dynsys)
        dsys = self.diff_dynsys(l)
        M = sp.Matrix(l, l, lambda i, j: sp.simplify(dsys[j][i % l].rhs))
        slow_mf = sp.Eq(M.det(), 0)
        return slow_mf

    def plot_slow_mf(self, params, save=False, filename="slow_mf.png", show=True):
        """Plot the slow manifold.

		Parameters
		----------
		params : dict
		    Numerical values for parameters.
		save : bool
		    Save file? Defaults to False.
		filename : str
		    Name of the file if saved.
		    (Default value = "slow_mf.png")
		show : bool
			Show the plot.
		    (Default value = True)

		Returns
		-------
		sympy.plotting.plot.backend instance.
		"""
        var, fun = self.variables()
        substitutes = zip(fun, var)
        expr = self.slow_mf().subs(substitutes)
        expr2 = expr.subs(params).simplify()
        print(expr2)
        plot = sp.plotting.plot_implicit(expr2, show=False)
        backend = plot.backend(plot)
        backend.process_series()
        if save == True:
            backend.fig.savefig(filename, dpi=300)
        if show:
            backend.show()
        return backend

    def plot_slow_mf_and_simulation(
        self,
        params,
        save=False,
        filename="slow_mf.png",
        numerics={"time": [0, 1, 100], "initial_condition": [1, 1]},
    ):
        """Plot the slow manifold and a simulation of the system for specified range and initial conditions.

		Parameters
		----------
		params : dict
		    Numerical values for parameters.
		save : bool
		    Save file? Defaults to False.
		filename : str
			Name of the file if saved. (Default value = "slow_mf.png")
		numerics : dict
		    Contains timerange and initial conditions for simulation. (Default value = {"time":[0)

		Returns
		-------
		`sympy.plotting.plot.backend` instance
		"""

        # Check if numerics dict is correct
        if not numerics["time"] or not numerics["initial_condition"]:
            print("Please give valid numerics dictionary.")
        else:
            timespan = numerics["time"]
            ic = numerics["initial_condition"]

        # Prepare simulation
        f = self.make_integrable(params)
        t = np.linspace(*timespan)
        sol = spi.odeint(f, ic, t)

        backend = self.plot_slow_mf(params, save=False, show=False)
        backend.ax.scatter(sol[:, 0], sol[:, 1], c="r")

        if save == True:
            backend.fig.savefig(filename, dpi=300)
        backend.show()

    def plot_slow_mf_and_simulation_3d(
        self,
        params,
        save=False,
        filename="slow_mf.png",
        numerics={"time": [0, 1, 100], "initial_condition": [1, 1, 1]},
        bbox=(-2.5, 2.5),
    ):
        """Plot the slow manifold of a three dimensional system and a simulation of the system for specified range and initial conditions.

		Parameters
		----------
		params : dict
		    Numerical values for parameters.
		save : bool
		    Save file? Defaults to False.
		filename : str
		    Name of the file if saved. (Default value = "slow_mf.png")
		numerics : dict
		    Contains timerange and initial conditions for simulation. (Default value = {"time":[0)

		"""
        from mpl_toolkits.mplot3d import Axes3D

        def plot_implicit(fn, bbox=(-2.5, 2.5)):
            """create a plot of an implicit function
			fn  ...implicit function (plot where fn==0)
			bbox ..the x,y,and z limits of plotted interval

			Parameters
			----------
			fn : function
			    
			bbox : tuple
				Axis range
			    (Default value = (-2.52.5)
			"""
            xmin, xmax, ymin, ymax, zmin, zmax = bbox * 3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            A = np.linspace(xmin, xmax, 100)  # resolution of the contour
            B = np.linspace(xmin, xmax, 15)  # number of slices
            A1, A2 = np.meshgrid(A, A)  # grid on which the contour is plotted

            for z in B:  # plot contours in the XY plane
                X, Y = A1, A2
                Z = fn(X, Y, z)
                cset = ax.contour(X, Y, Z + z, [z], zdir="z")
                # [z] defines the only level to plot for this contour for this value of z

            for y in B:  # plot contours in the XZ plane
                X, Z = A1, A2
                Y = fn(X, y, Z)
                cset = ax.contour(X, Y + y, Z, [y], zdir="y")

            for x in B:  # plot contours in the YZ plane
                Y, Z = A1, A2
                X = fn(x, Y, Z)
                cset = ax.contour(X + x, Y, Z, [x], zdir="x")

            # must set plot limits because the contour will likely extend
            # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
            # to encompass all values in the contour.
            ax.set_zlim3d(zmin, zmax)
            ax.set_xlim3d(xmin, xmax)
            ax.set_ylim3d(ymin, ymax)

            return fig, ax

        var, fun = self.variables()
        substitutes = zip(fun, var)
        fn = self.slow_mf().lhs.subs(params)
        f = sp.lambdify(["x1_0", "x2_0", "x3_0"], fn.subs(substitutes))
        fig, ax = plot_implicit(f, bbox=bbox)

        f2 = self.make_integrable(params)
        t = np.linspace(*numerics["time"])
        y0 = numerics["initial_condition"]
        sol = spi.odeint(f2, y0, t)

        ax.scatter(sol.T[0], sol.T[1], sol.T[2])
        if save == True:
            fig.savefig(filename)
        plt.show()



	def lie_derivative(self):
		"""
		Compute the Lie derivative of the given equation along the given vector field. 
		In practice this will be used for the Lie derivative of the slow manifold equation.
		"""
		#TODO Das nochmal verstehen...
		equation = self.slow_mf()
		dynsys = self.ode2dynsys()
		var = self.variables()
		variables = [i for i in var[1]] #TODO So richtig sauber sind das und die nächsten Zeilen nicht.
		vector_field = [i for i in var[1]]
		del variables[-1]
		del vector_field[0]

		lie_deriv = 0

		for var, vec in zip(variables, vector_field):
			lie_deriv += sp.diff(equation.lhs, var) * vec
		lie_deriv = lie_deriv.subs(vector_field[-1], dynsys[-1].rhs)
		lie_deriv.simplify()

		return lie_deriv



	def decompose_lie_derivative(self):
		"""
		Compute the Lie derivative of the given equation along the given vector field. 
		In practice this will be used for the Lie derivative of the slow manifold equation.
    
		:param equation: sympy.equation the left hand side of which the Lie derivative is to be taken of.
		:param dynsys: Dynamical system equations to be substituted.
		:param variables: Independent variables to calculate partial derivatives.
		:param vector_field: Vector field along which the Lie derivative is to be taken.
		"""

		#TODO Das nochmal verstehen...
		#TODO Not really a method yet but should work as a standalone function

		equation = self.slow_mf()

#		variables = [i for i in var[1]]
#		vector_field = [i for i in var[1]]
		#print(var)
		#print(var[1])
#		del variables[-1]
#		print(variables)
#		del vector_field[0]
#		print(vector_field[-1])

		#TODO Check shape of vector field
#		lie_deriv = 0
		Phi = sp.Symbol(r'\Phi')
		Psi = sp.Symbol(r'\Psi')
		lie_deriv = self.lie_derivative()
		#TODO Eigentlich sind nur die nächsten vier Zeilen lie-deriv und sollten funktional vom Rest getrennt werden
#		for var, vec in zip(variables, vector_field):
#			lie_deriv += sp.diff(equation.lhs, var) * vec
#		lie_deriv = lie_deriv.subs(vector_field[-1], dynsys[-1].rhs) #TODO Substitue dynamical equations!
#		lie_deriv.simplify()
#		print(lie_deriv)
		k, Psi = sp.div(lie_deriv, equation.lhs)
		#lie_deriv = q*Phi + r
		#r_plot = r.subs({'p1_1':6,'p1_2':1,variables[0]:'x', variables[1]: 'y'})
		#print(r_plot)
		#plot = sp.plotting.plot_implicit(r_plot,show=True)
		return lie_deriv, k, Psi



def make_callable(expression):
    """
	Make a sympy expression callable with numeric values.

    Parameters
    ----------
    expression : `sympy.Expression`
        

    Returns
    -------
	callable
    """
    variables = sorted(expression.free_symbols, key=lambda symbol: symbol.name)
    print(variables)
    f = sp.lambdify(variables, expression)
    return f
