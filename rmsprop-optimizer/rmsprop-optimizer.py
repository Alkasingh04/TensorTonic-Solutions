import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    """
    Perform one RMSProp update step.
    
    Parameters
    ----------
    w : np.array
        Current weights
    g : np.array
        Gradient of loss w.r.t weights
    s : np.array
        Running average of squared gradients
    lr : float
        Learning rate
    beta : float
        Decay rate
    eps : float
        Small value for numerical stability
    """
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)
    # Update running average of squared gradients
    s = beta * s + (1 - beta) * (g ** 2)

    # Update weights
    w = w - lr * g / (np.sqrt(s) + eps)

    return w, s