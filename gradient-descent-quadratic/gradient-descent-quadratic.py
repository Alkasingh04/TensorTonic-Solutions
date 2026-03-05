def gradient_descent_quadratic(a, b, c, x0, lr, iterations):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for i in range(iterations):
        grad = 2*a*x + b
        x = x - lr * grad
        
        print(f"Iter {i+1}: x = {x:.4f}")
        
    return x