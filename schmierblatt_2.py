from data_simulate import generate_laser_projection_on_rotated_plane, project_points_to_cameras, plot_2d_points_pair
from calc_3d import triangulate_3d
from file_io.plotting import plot_3d_points



d = 10
n = 8
alpha = 1/3

camera_stats = {
    'f': 0.05,               # Focal length of the camera in meters
    'pixel_size': 2.74e-6,   # Size of a single pixel in meters (2.74 µm)
    'resolution': (4096, 3000)  # Resolution of the camera in pixels (width, height)
}

laser_points = generate_laser_projection_on_rotated_plane(d, n, alpha, beta=None, angle=(0, 0))

plot_3d_points(laser_points)

cam1_points, cam2_points = project_points_to_cameras(laser_points, camera_stats)

print(laser_points.shape)
#print(cam1_points)
print(cam2_points.shape)

plot_2d_points_pair(cam1_points, cam2_points)

points_3d = triangulate_3d(cam1_points, cam2_points, camera_stats)

plot_3d_points(points_3d)
    

