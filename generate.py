import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

class Generate_points:
    def __init__(self, f, u, number_of_nodes, number_of_datapoints, x_boundaries, key_number, evaluation_points, noise, dimension = 1):
        self.f = f
        self.u = u
        self.x_boundaries = x_boundaries
        self.evaluation_points = evaluation_points
        self.K = number_of_nodes
        self.J = number_of_datapoints
        self.key = jax.random.PRNGKey(key_number)
        self.noise = noise
        self.dimension = dimension

    def generate(self):
        self.key, self.xkey, self.noisekey, self.wkey, self.bkey, self.ykey = jax.random.split(self.key, 6)
        x = jax.random.uniform(self.xkey, (self.dimension, self.J), minval = self.x_boundaries[0], maxval = self.x_boundaries[1])
        self.epsilon = jax.random.uniform(self.ykey, (1, self.J), minval = self.x_boundaries[0], maxval = self.x_boundaries[1]) * self.noise
        self.x = jnp.sort(x)
        self.x_eval = jnp.linspace(0, 2 * jnp.pi, self.evaluation_points)[None, :] #Dimension = 1 här
        self.w = jax.random.normal(self.wkey, (1, self.K)) * 2 #1 x K
        self.b = jax.random.normal(self.bkey, (self.K, 1)) #1 x K
        self.y = jax.vmap(self.f)(self.x.flatten())[None, :] + self.epsilon
        self.y_evaluate = self.u(self.x_eval) #Without noise
        return self.w, self.b, self.x, self.y, self.x_eval, self.y_evaluate
        
        
        
        
        
        