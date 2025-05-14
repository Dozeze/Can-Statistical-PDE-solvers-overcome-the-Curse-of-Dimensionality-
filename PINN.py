import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

def layer1(w, x):
    return jnp.dot(x.T, w) # J x K
    
def activation(w_dot_x):
    return jnp.exp(1j * w_dot_x) # K x J
    
def layer2(exp_w_dot_x, b):
    return jnp.dot(exp_w_dot_x, b) # 1 x J

def forward(w, b, x):
    return layer2(activation(layer1(w,x)), b).real #1 x J

def minimi(w,x,b,f,reg):
    "Returns mean(u^2 - u'^2) + lambda * |b_k|^2"
    fx = jax.vmap(f)(x).T
    Aw = activation(layer1(w,x))
    w_re = w.flatten()
    Dw = 1j * jnp.diag((w_re)) #(K x K)
    I = jnp.eye(len(w_re)) #(K x K)
    t1 = jnp.dot(Aw, b)
    t2 = jnp.dot(jnp.dot(Aw, Dw),b)
    t3 = jnp.dot(Aw, b) * fx #100 x 1, 1 x 30 => 100 x 30
    F = 0.5 * (jnp.abs(t1) + jnp.abs(t2)) - jnp.abs(t3)
    term1 = jnp.mean(F)
    term2 = reg * jnp.linalg.norm(b)**2
    return term1 + term2

def evaluationerror(w,x,b,y):
    "Returns the mean square error"
    return jnp.mean((forward(w,b,x) - y.T)**2)

@jax.jit
def get_b(w,x,fx,reg):
    "Returns the solution to (A^*A + D^*A^*AD)b = Af(x)^T"
    w_re = w.flatten() #(, K)
    Dw = 1j * jnp.diag((w_re)) #(K x K)
    Aw = activation(layer1(w,x)) #(J x K)
    Dw_sword = jnp.conjugate(Dw).T #(K x K)
    Aw_sword = jnp.conjugate(Aw).T #(K x J)
    I = jnp.eye(len(w_re)) #(K x K)
    A = jnp.dot(Aw_sword, Aw) + jnp.dot(jnp.dot(Dw_sword, Aw_sword), jnp.dot(Aw, Dw)) + reg * I
    b = jnp.dot(Aw_sword, fx)
    return jnp.linalg.solve(A,b)
    

@jax.jit
def get_p(w, b, alpha):
    "Returns optimal sample with scaling"
    #print(jnp.abs(w))
    #print(jnp.abs(b.T))
    p_w = jnp.sqrt(1 + alpha * jnp.abs(w)**2) * jnp.abs(b.T) #(1 x K)
    p_w = p_w / jnp.sum(p_w) #Testa skala om så vi får sannolikhetsfunktion
    return p_w
    
@jax.jit
def get_w(key, p_w, w):
    "Resamples w from optimal sampling"
    p_w = p_w.flatten()
    w = w.flatten() #(, K) (1,K)
    w_new = jax.random.choice(key, a = w, shape = w.shape, replace = True, p = p_w)[None, :] # Kommer bli problem
    return w_new # (1 x K)

def update(w, b, f, x, epochs, key, scale, reg, dim, delta, x_eval, y_eval, K):
    "Training neural network"
    err_list = []
    #Y = f(x).T
    Y = jax.vmap(f)(x.flatten())[None, :].T
    #Y = jax.vmap(f)(x)[None, :]
    for epoch in range(1, epochs + 1):
        if (epoch % 10) == 0:
            error = evaluationerror(w, x_eval, b, y_eval)
            #error2 = minimi(w,x,b,f,reg)
            #print("epoch: ", epoch, "Mean square error:", error)#, "Other error term:", error2)
            err_list.append(error)
        key, subkey = jax.random.split(key, 2)
        zeta = jax.random.normal(subkey, (dim, K))
        w += delta * zeta
        b = get_b(w,x,Y,reg)
        p_w = get_p(w, b, scale)
        key, subkey = jax.random.split(key,2)
        w = get_w(subkey, p_w, w)
        b = get_b(w,x,Y,reg)
    b = get_b(w,x,Y,reg)
    return w, b, err_list