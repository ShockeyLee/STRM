import airsim
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatial_intersection import spatial_intersection_robust, collinearity_equations

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        q: Quaternion (w, x, y, z)
        
    Returns:
        roll, pitch, yaw in radians
    """
    # Extract the values from q
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def airsim_camera_to_photogrammetry_params(pose, focal_length, image_width, image_height):
    """
    Convert AirSim camera pose to photogrammetry parameters
    
    Args:
        pose: AirSim camera pose
        focal_length: focal length in pixels
        image_width, image_height: image dimensions in pixels
        
    Returns:
        XL, YL, ZL, omega, phi, kappa, x0, y0, f: camera parameters for photogrammetry
    """
    # Extract position
    position = pose.position
    XL = position.x_val
    YL = position.y_val
    ZL = position.z_val
    
    # Convert orientation to Euler angles
    orientation = pose.orientation
    q = [orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]
    roll, pitch, yaw = quaternion_to_euler(q)
    
    # In photogrammetry convention:
    # omega is rotation around x-axis (roll)
    # phi is rotation around y-axis (pitch)
    # kappa is rotation around z-axis (yaw)
    # Note: AirSim and photogrammetry may have different coordinate systems,
    # so adjust accordingly if needed
    omega = roll
    phi = pitch
    kappa = yaw
    
    # Principal point (assume at center of image)
    x0 = image_width / 2
    y0 = image_height / 2
    
    # Focal length in pixel units
    f = focal_length
    
    return XL, YL, ZL, omega, phi, kappa, x0, y0, f

def pixel_to_mm(pixel_coords, image_width, image_height, sensor_width, sensor_height):
    """
    Convert pixel coordinates to mm on the sensor
    
    Args:
        pixel_coords: (x, y) coordinates in pixels
        image_width, image_height: image dimensions in pixels
        sensor_width, sensor_height: sensor dimensions in mm
        
    Returns:
        x, y coordinates in mm on the sensor
    """
    x_pixel, y_pixel = pixel_coords
    
    # Convert to mm (origin at center of image)
    x_mm = (x_pixel - image_width / 2) * sensor_width / image_width
    y_mm = (y_pixel - image_height / 2) * sensor_height / image_height
    
    return x_mm, y_mm

def run_validation():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Reset and enable API control
    client.reset()
    client.enableApiControl(True)
    
    # Define camera parameters (these should match your AirSim settings)
    image_width = 1280
    image_height = 720
    fov_degrees = 90
    # Calculate focal length in pixels
    focal_length_pixels = (image_width / 2) / np.tan(np.radians(fov_degrees / 2))
    # Assume a 35mm sensor for this example
    sensor_width = 36  # mm
    sensor_height = 24  # mm
    
    # Take off
    client.takeoffAsync().join()
    
    # Move to first position
    client.moveToPositionAsync(0, 0, -10, 5).join()  # x, y, z, velocity
    
    # Define a target object position
    target_position = np.array([10, 5, -5])  # Some point in the world
    
    # List to store camera parameters and image points
    camera_params = []
    image_points = []
    camera_positions = []
    
    # Capture from multiple positions
    positions = [
        (0, 0, -10),     # Position 1
        (5, 0, -8),      # Position 2
        (8, 4, -7),      # Position 3
        (3, 7, -9)       # Position 4
    ]
    
    for i, pos in enumerate(positions):
        # Move drone to the position
        x, y, z = pos
        client.moveToPositionAsync(x, y, z, 5).join()
        time.sleep(1)  # Give time to stabilize
        
        # Get current camera pose
        pose = client.simGetCameraPose("0")
        
        # Get image
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Display image with text
        cv2.imshow(f"Position {i+1}", img_bgr)
        cv2.waitKey(1)
        
        # Get camera parameters
        camera_param = airsim_camera_to_photogrammetry_params(
            pose, focal_length_pixels, image_width, image_height)
        camera_params.append(camera_param)
        
        # Store camera position for visualization
        camera_positions.append((camera_param[0], camera_param[1], camera_param[2]))
        
        # Project the target point onto the image
        # We need to convert from AirSim coordinate system to camera coordinate system
        position = pose.position
        XL, YL, ZL = position.x_val, position.y_val, position.z_val
        
        # Calculate the target's coordinates in camera frame
        camera_point = target_position - np.array([XL, YL, ZL])
        
        # Use collinearity equations to get image coordinates
        X, Y, Z = target_position
        XL, YL, ZL, omega, phi, kappa, x0, y0, f = camera_param
        
        # Get image coordinates in pixels
        x_pixel, y_pixel = collinearity_equations(X, Y, Z, XL, YL, ZL, omega, phi, kappa, 0, 0, focal_length_pixels)
        
        # Convert to center-origin pixels
        x_pixel += image_width / 2
        y_pixel += image_height / 2
        
        # Convert pixel coordinates to mm for photogrammetry
        x_mm, y_mm = pixel_to_mm((x_pixel, y_pixel), image_width, image_height, sensor_width, sensor_height)
        
        # Add small random noise to simulate measurement error
        x_mm += np.random.normal(0, 0.01)  # 0.01mm standard deviation
        y_mm += np.random.normal(0, 0.01)
        
        # Add to image points list
        image_points.append((x_mm, y_mm))
        
        # Draw the point on the image
        cv2.circle(img_bgr, (int(x_pixel), int(y_pixel)), 10, (0, 255, 0), -1)
        cv2.imshow(f"Position {i+1} with Target", img_bgr)
        cv2.waitKey(0)
    
    # Close any remaining windows
    cv2.destroyAllWindows()
    
    # Perform spatial intersection
    estimated_point = spatial_intersection_robust(image_points, camera_params)
    
    print(f"True target position: {target_position}")
    print(f"Estimated position: {estimated_point}")
    print(f"Error: {np.linalg.norm(target_position - estimated_point):.3f} units")
    
    # Visualize the results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    camera_positions = np.array(camera_positions)
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
               color='blue', s=100, label='Camera Positions')
    
    # Plot true target position
    ax.scatter(target_position[0], target_position[1], target_position[2], 
               color='red', s=200, label='True Target Position')
    
    # Plot estimated target position
    ax.scatter(estimated_point[0], estimated_point[1], estimated_point[2], 
               color='green', s=200, label='Estimated Target Position')
    
    # Draw lines from cameras to the estimated point
    for cam_pos in camera_positions:
        ax.plot([cam_pos[0], estimated_point[0]], 
                [cam_pos[1], estimated_point[1]], 
                [cam_pos[2], estimated_point[2]], 'k--', alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spatial Intersection Validation using AirSim')
    ax.legend()
    
    plt.show()
    
    # Return to start and land
    client.moveToPositionAsync(0, 0, -10, 5).join()
    client.landAsync().join()
    client.enableApiControl(False)

if __name__ == "__main__":
    try:
        run_validation()
    except Exception as e:
        print(f"Error: {e}")
        # Ensure drone is reset in case of error
        client = airsim.MultirotorClient()
        client.reset()
        client.enableApiControl(False)