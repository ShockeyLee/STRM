import airsim
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatial_intersection import spatial_intersection_robust

class PointSelector:
    def __init__(self, window_name):
        self.window_name = window_name
        self.points = []
        self.current_point = None
        self.image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record the point
            self.current_point = (x, y)
            # Draw a circle at the selected point
            if self.image is not None:
                img_copy = self.image.copy()
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(img_copy, f"Point: ({x}, {y})", (x + 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show live coordinates
            if self.image is not None:
                img_copy = self.image.copy()
                cv2.putText(img_copy, f"({x}, {y})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if self.current_point:
                    cv2.circle(img_copy, self.current_point, 3, (0, 255, 0), -1)
                cv2.imshow(self.window_name, img_copy)

def select_points_on_images(images, camera_infos):
    """
    Interactive point selection on multiple images
    
    Args:
        images: List of images
        camera_infos: List of camera information
        
    Returns:
        image_points: List of selected points in mm coordinates
    """
    image_points = []
    image_pixel_points = []  # Store pixel coordinates for visualization
    
    print("\n=== Point Selection Instructions ===")
    print("1. Click to select a point on each image")
    print("2. Press 'SPACE' to confirm point selection")
    print("3. Press 'r' to retry current image")
    print("4. Press 'q' to quit")
    print("5. Select the SAME point in all images")
    print("=====================================\n")

    for i, (img, camera_info) in enumerate(zip(images, camera_infos)):
        window_name = f"Image {i+1} - Select Point"
        selector = PointSelector(window_name)
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, selector.mouse_callback)
        
        while True:
            selector.image = img.copy()
            # Show previous selections if any
            for j, point in enumerate(image_pixel_points):
                cv2.circle(selector.image, point, 3, (0, 0, 255), -1)
                cv2.putText(selector.image, f"Image {j+1}", (point[0] + 10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(window_name, selector.image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to confirm selection
                if selector.current_point is not None:
                    # Convert pixel coordinates to mm
                    x_pixel, y_pixel = selector.current_point
                    x_mm, y_mm = pixel_to_mm((x_pixel, y_pixel), camera_info)
                    image_points.append((x_mm, y_mm))
                    image_pixel_points.append(selector.current_point)
                    print(f"Point selected on Image {i+1}: Pixel({x_pixel}, {y_pixel}), mm({x_mm:.2f}, {y_mm:.2f})")
                    break
            elif key == ord('r'):  # r to retry
                selector.current_point = None
                print(f"Retrying point selection for Image {i+1}")
            elif key == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyWindow(window_name)
    
    return image_points, image_pixel_points

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
    hfov = camera_info.fov * np.pi / 180.0
    focal_length_pixels = camera_info.proj_mat[0][0]
    
    # Calculate sensor width in mm
    sensor_width = 2 * focal_length_pixels * np.tan(hfov / 2)
    sensor_height = sensor_width * (height / width)
    
    # Convert to mm (origin at center of image)
    x_mm = (x_pixel - width / 2) * sensor_width / width
    y_mm = (y_pixel - height / 2) * sensor_height / height
    
    return x_mm, y_mm

def run_interactive_validation():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Reset and enable API control
    client.reset()
    client.enableApiControl(True)
    
    # Take off
    client.takeoffAsync().join()
    
    # List to store images, camera parameters and positions
    images = []
    camera_params = []
    camera_positions = []
    camera_infos = []
    
    # Capture from multiple positions
    positions = [
        (0, 0, -10),     # Position 1
        (5, 0, -8),      # Position 2
        (8, 4, -7),      # Position 3
        (3, 7, -9)       # Position 4
    ]
    
    camera_name = "0"
    
    print("\nCapturing images from multiple positions...")
    for i, pos in enumerate(positions):
        # Move drone to the position
        x, y, z = pos
        client.moveToPositionAsync(x, y, z, 5).join()
        time.sleep(1)  # Give time to stabilize
        
        # Get current camera pose and info
        pose = client.simGetCameraPose(camera_name)
        camera_info = client.simGetCameraInfo(camera_name)
        
        # Get image
        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene)])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Store data
        images.append(img_bgr)
        camera_infos.append(camera_info)
        
        # Convert camera parameters
        XL = pose.position.x_val
        YL = pose.position.y_val
        ZL = pose.position.z_val
        
        # Convert orientation to Euler angles
        q = [pose.orientation.w_val, pose.orientation.x_val, 
             pose.orientation.y_val, pose.orientation.z_val]
        roll, pitch, yaw = quaternion_to_euler(q)
        
        # Get camera intrinsics
        x0 = camera_info.proj_mat[0][2]  # cx
        y0 = camera_info.proj_mat[1][2]  # cy
        f = (camera_info.proj_mat[0][0] + camera_info.proj_mat[1][1]) / 2  # focal length
        
        camera_params.append((XL, YL, ZL, roll, pitch, yaw, x0, y0, f))
        camera_positions.append((XL, YL, ZL))
        
        print(f"Position {i+1} captured: ({XL:.2f}, {YL:.2f}, {ZL:.2f})")
    
    # Interactive point selection
    print("\nStarting interactive point selection...")
    result = select_points_on_images(images, camera_infos)
    
    if result is None:
        print("Point selection cancelled")
        client.landAsync().join()
        client.enableApiControl(False)
        return
    
    image_points, image_pixel_points = result
    
    # Perform spatial intersection
    estimated_point = spatial_intersection_robust(image_points, camera_params)
    
    print("\nResults:")
    print(f"Estimated 3D position: ({estimated_point[0]:.3f}, {estimated_point[1]:.3f}, {estimated_point[2]:.3f})")
    
    # Visualize the results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    camera_positions = np.array(camera_positions)
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
               color='blue', s=100, label='Camera Positions')
    
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
    ax.set_title('Spatial Intersection Result')
    ax.legend()
    
    plt.show()
    
    # Return to start and land
    client.moveToPositionAsync(0, 0, -10, 5).join()
    client.landAsync().join()
    client.enableApiControl(False)

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

if __name__ == "__main__":
    try:
        run_interactive_validation()
    except Exception as e:
        print(f"Error: {e}")
        # Ensure drone is reset in case of error
        try:
            client = airsim.MultirotorClient()
            client.reset()
            client.enableApiControl(False)
        except:
            pass