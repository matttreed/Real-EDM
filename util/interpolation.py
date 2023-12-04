import numpy as np

def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    print(so)
    if so == 0:
        return (1.0 - val) * low + val * high  # Lerp if points are very close
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def spherical_blend(points, weights):
    if len(points) != len(weights):
        raise ValueError("Number of points and weights must be equal")
    
    # Normalize weights to sum to 1
    weights /= np.sum(weights)

    # Start with the first point
    blend = points[0]

    for i in range(1, len(points)):
        # Blend between the current blend and the next point
        blend = slerp(weights[i], blend, points[i])

    return blend