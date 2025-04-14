import numpy as np
from scipy.optimize import least_squares

def rotation_matrix(omega, phi, kappa):
    """
    Calculate rotation matrix from rotation angles omega, phi, kappa
    
    Args:
        omega: rotation angle around x-axis (rad)
        phi: rotation angle around y-axis (rad)
        kappa: rotation angle around z-axis (rad)
        
    Returns:
        R: 3x3 rotation matrix
    """
    # Rotation matrix
    R = np.zeros((3, 3))
    
    R[0, 0] = np.cos(phi) * np.cos(kappa)
    R[0, 1] = np.sin(omega) * np.sin(phi) * np.cos(kappa) + np.cos(omega) * np.sin(kappa)
    R[0, 2] = -np.cos(omega) * np.sin(phi) * np.cos(kappa) + np.sin(omega) * np.sin(kappa)
    
    R[1, 0] = -np.cos(phi) * np.sin(kappa)
    R[1, 1] = -np.sin(omega) * np.sin(phi) * np.sin(kappa) + np.cos(omega) * np.cos(kappa)
    R[1, 2] = np.cos(omega) * np.sin(phi) * np.sin(kappa) + np.sin(omega) * np.cos(kappa)
    
    R[2, 0] = np.sin(phi)
    R[2, 1] = -np.sin(omega) * np.cos(phi)
    R[2, 2] = np.cos(omega) * np.cos(phi)
    
    return R

def collinearity_equations(X, Y, Z, XL, YL, ZL, omega, phi, kappa, x0, y0, f):
    """
    Calculate image coordinates using collinearity equations
    
    Args:
        X, Y, Z: object point coordinates
        XL, YL, ZL: perspective center coordinates
        omega, phi, kappa: rotation angles
        x0, y0: principal point coordinates
        f: focal length
        
    Returns:
        x, y: image coordinates
    """
    # Rotation matrix
    R = rotation_matrix(omega, phi, kappa)
    
    # Coordinate differences
    dX = X - XL
    dY = Y - YL
    dZ = Z - ZL
    
    # Denominator
    den = R[2, 0] * dX + R[2, 1] * dY + R[2, 2] * dZ
    
    # Image coordinates
    x = x0 - f * (R[0, 0] * dX + R[0, 1] * dY + R[0, 2] * dZ) / den
    y = y0 - f * (R[1, 0] * dX + R[1, 1] * dY + R[1, 2] * dZ) / den
    
    return x, y

def spatial_intersection(image_points, camera_params):
    """
    Compute the 3D coordinates of a point using spatial intersection (point projection coefficient method)
    
    Args:
        image_points: list of image coordinates [(x1, y1), (x2, y2), ...] for each image
        camera_params: list of camera parameters [(XL, YL, ZL, omega, phi, kappa, x0, y0, f), ...]
                      for each image
        
    Returns:
        X, Y, Z: 3D coordinates of the object point
    """
    # Number of images
    n_images = len(image_points)
    
    # Check if we have at least two images
    if n_images < 2:
        raise ValueError("At least two images are required for spatial intersection")
    
    # Initialize matrices
    A = np.zeros((2 * n_images, 3))
    b = np.zeros(2 * n_images)
    
    # For each image
    for i in range(n_images):
        # Extract image coordinates
        x, y = image_points[i]
        
        # Extract camera parameters
        XL, YL, ZL, omega, phi, kappa, x0, y0, f = camera_params[i]
        
        # Calculate rotation matrix
        R = rotation_matrix(omega, phi, kappa)
        
        # Calculate coefficients
        a1 = (x - x0) * R[2, 0] + f * R[0, 0]
        a2 = (x - x0) * R[2, 1] + f * R[0, 1]
        a3 = (x - x0) * R[2, 2] + f * R[0, 2]
        
        b1 = (y - y0) * R[2, 0] + f * R[1, 0]
        b2 = (y - y0) * R[2, 1] + f * R[1, 1]
        b3 = (y - y0) * R[2, 2] + f * R[1, 2]
        
        # Fill matrices
        A[2*i, 0] = a1
        A[2*i, 1] = a2
        A[2*i, 2] = a3
        b[2*i] = a1 * XL + a2 * YL + a3 * ZL
        
        A[2*i+1, 0] = b1
        A[2*i+1, 1] = b2
        A[2*i+1, 2] = b3
        b[2*i+1] = b1 * XL + b2 * YL + b3 * ZL
    
    # Solve system using least squares
    X, Y, Z = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return X, Y, Z

def spatial_intersection_error(point, image_points, camera_params):
    """
    Compute error for spatial intersection
    
    Args:
        point: 3D coordinates [X, Y, Z]
        image_points: list of image coordinates [(x1, y1), (x2, y2), ...] for each image
        camera_params: list of camera parameters [(XL, YL, ZL, omega, phi, kappa, x0, y0, f), ...]
                      for each image
        
    Returns:
        error: array of residuals
    """
    X, Y, Z = point
    n_images = len(image_points)
    error = np.zeros(2 * n_images)
    
    for i in range(n_images):
        x_obs, y_obs = image_points[i]
        XL, YL, ZL, omega, phi, kappa, x0, y0, f = camera_params[i]
        
        x_calc, y_calc = collinearity_equations(X, Y, Z, XL, YL, ZL, omega, phi, kappa, x0, y0, f)
        
        error[2*i] = x_obs - x_calc
        error[2*i+1] = y_obs - y_calc
    
    return error

def spatial_intersection_robust(image_points, camera_params, initial_guess=None):
    """
    Robust spatial intersection using nonlinear least squares
    
    Args:
        image_points: list of image coordinates [(x1, y1), (x2, y2), ...] for each image
        camera_params: list of camera parameters [(XL, YL, ZL, omega, phi, kappa, x0, y0, f), ...]
                      for each image
        initial_guess: initial guess for 3D coordinates [X, Y, Z]
        
    Returns:
        X, Y, Z: 3D coordinates of the object point
    """
    # Get initial approximation using point projection coefficient method
    if initial_guess is None:
        initial_guess = spatial_intersection(image_points, camera_params)
    
    # Refine using least squares
    result = least_squares(spatial_intersection_error, initial_guess, 
                          args=(image_points, camera_params))
    
    X, Y, Z = result.x
    return X, Y, Z

# Example usage
if __name__ == "__main__":
    # Example data
    # Image coordinates for two images (x, y) in mm
    image_points = [
        (-10.5, 8.2),  # Image 1
        (12.3, 6.7)    # Image 2
    ]
    
    # Camera parameters (XL, YL, ZL, omega, phi, kappa, x0, y0, f) 
    # XL, YL, ZL in meters, angles in radians, x0, y0, f in mm
    camera_params = [
        (1000.0, 2000.0, 1500.0, 0.01, 0.02, 0.03, 0.0, 0.0, 152.0),  # Camera 1
        (1100.0, 2050.0, 1550.0, 0.02, 0.01, 0.04, 0.0, 0.0, 152.0)   # Camera 2
    ]
    
    # Calculate 3D coordinates
    X, Y, Z = spatial_intersection_robust(image_points, camera_params)
    
    print(f"3D coordinates (X, Y, Z): ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")