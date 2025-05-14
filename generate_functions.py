import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

def generate_function(u):
    phi = lambda x : -4 * jnp.cos(x/4)
    psi = lambda x : -4 * jnp.cos(-x/4 + 3 * jnp.pi / 2)
    der_u = jax.grad(u)
    term1 = lambda t : phi(t) * der_u(2 * jnp.pi)
    term2 = lambda t : psi(t) * der_u(0.0)
    u_new = lambda t : u(t) - term1(t) - term2(t)
    return u_new

def generate_f(u):
    u_tt = jax.grad(jax.grad(u))
    f = lambda t : u(t) - u_tt(t)
    #u_tt = jax.vmap(u_tt)
    return f #u_tt