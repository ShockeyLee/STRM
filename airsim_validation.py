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

def airsim_camera_to_photogrammetry_params(pose, camera_info):
    """
    Convert AirSim camera pose and info to photogrammetry parameters
    
    Args:
        pose: AirSim camera pose
        camera_info: AirSim camera information
        
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
    
    # Get camera parameters from camera_info
    # Principal point (assume at center of image)
    x0 = camera_info.proj_mat[0][2]  # cx from projection matrix
    y0 = camera_info.proj_mat[1][2]  # cy from projection matrix
    
    # Focal length in pixel units
    fx = camera_info.proj_mat[0][0]
    fy = camera_info.proj_mat[1][1]
    # Use average of fx and fy for simplicity
    f = (fx + fy) / 2
    
    return XL, YL, ZL, omega, phi, kappa, x0, y0, f

def pixel_to_mm(pixel_coords, camera_info):
    """
    Convert pixel coordinates to mm on the sensor
    
    Args:
        pixel_coords: (x, y) coordinates in pixels
        camera_info: AirSim camera information
        
    Returns:
        x, y coordinates in mm on the sensor
    """
    x_pixel, y_pixel = pixel_coords
    
    # Get image dimensions
    width = camera_info.width
    height = camera_info.height
    
    # Calculate sensor dimensions based on FOV and focal length
    # Note: This is an approximation. In a real camera, we would have actual sensor dimensions
    # Horizontal FOV in radians
    hfov = camera_info.fov * np.pi / 180.0
    
    # Using the focal length from the projection matrix
    focal_length_pixels = camera_info.proj_mat[0][0]
    
    # Calculate sensor width in mm (assuming standard 35mm equivalent conventions)
    # This is a simplified model - in reality, sensor size would be a known parameter
    sensor_width = 2 * focal_length_pixels * np.tan(hfov / 2)
    sensor_height = sensor_width * (height / width)  # Maintain aspect ratio
    
    # Convert to mm (origin at center of image)
    x_mm = (x_pixel - width / 2) * sensor_width / width
    y_mm = (y_pixel - height / 2) * sensor_height / height
    
    return x_mm, y_mm

def run_validation():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Reset and enable API control
    client.reset()
    client.enableApiControl(True)
    
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
    
    # Camera name - use "0" for the default camera
    camera_name = "0"
    
    for i, pos in enumerate(positions):
        # Move drone to the position
        x, y, z = pos
        client.moveToPositionAsync(x, y, z, 5).join()
        time.sleep(1)  # Give time to stabilize
        
        # Get current camera pose
        pose = client.simGetCameraPose(camera_name)
        
        # Get camera info - this contains intrinsic parameters
        camera_info = client.simGetCameraInfo(camera_name)
        
        # Print camera info for debugging
        print(f"Camera {i+1} Info:")
        print(f"  FOV: {camera_info.fov} degrees")
        print(f"  Projection Matrix: \n{np.array(camera_info.proj_mat)}")
        print(f"  Image Size: {camera_info.width}x{camera_info.height}")
        
        # Get image
        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene)])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Display image with text
        cv2.imshow(f"Position {i+1}", img_bgr)
        cv2.waitKey(1)
        
        # Get camera parameters using camera_info
        camera_param = airsim_camera_to_photogrammetry_params(pose, camera_info)
        camera_params.append(camera_param)
        
        # Store camera position for visualization
        camera_positions.append((camera_param[0], camera_param[1], camera_param[2]))
        
        # Project the target point onto the image using collinearity equations
        X, Y, Z = target_position
        XL, YL, ZL, omega, phi, kappa, x0, y0, f = camera_param
        
        # Get image coordinates in pixels (relative to principal point)
        x_pixel_rel, y_pixel_rel = collinearity_equations(X, Y, Z, XL, YL, ZL, omega, phi, kappa, 0, 0, f)
        
        # Convert to image coordinates (origin at top-left)
        x_pixel = x_pixel_rel + x0
        y_pixel = y_pixel_rel + y0
        
        # Convert pixel coordinates to mm for photogrammetry
        x_mm, y_mm = pixel_to_mm((x_pixel, y_pixel), camera_info)
        
        # Add small random noise to simulate measurement error
        x_mm += np.random.normal(0, 0.01)  # 0.01mm standard deviation
        y_mm += np.random.normal(0, 0.01)
        
        # Add to image points list
        image_points.append((x_mm, y_mm))
        
        # Draw the point on the image
        cv2.circle(img_bgr, (int(x_pixel), int(y_pixel)), 10, (0, 255, 0), -1)
        cv2.putText(img_bgr, f"Target ({int(x_pixel)}, {int(y_pixel)})", 
                   (int(x_pixel) + 15, int(y_pixel)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(f"Position {i+1} with Target", img_bgr)
        cv2.waitKey(0)
    
    # Close any remaining windows
    cv2.destroyAllWindows()
    
    # Perform spatial intersection
    estimated_point = spatial_intersection_robust(image_points, camera_params)
    
    print("\nResults:")
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
        try:
            client = airsim.MultirotorClient()
            client.reset()
            client.enableApiControl(False)
        except:
            pass