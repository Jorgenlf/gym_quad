import numpy as np
from vispy import app, gloo, scene

import gym_quad.utils.geomutils as geom

class EnvironmentVisualizer(app.Canvas):
    def __init__(self, obstacles, quadcopter_position, quadcopter_attitude):
        app.Canvas.__init__(self, size=(800, 800), keys='interactive')
        self.obstacles = obstacles
        self.quadcopter_position = quadcopter_position
        self.quadcopter_attitude = quadcopter_attitude
        self.waypoints_visual = None
        self.create_visuals()

    def create_visuals(self):
        self.scene = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.scene.central_widget.add_view()
        self.grid = scene.visuals.GridLines(parent=self.view.scene)

        # Create axis visuals
        self.axis_x = scene.visuals.Line(pos=np.array([[0, 0, 0], [20, 0, 0]]), color='red', parent=self.view.scene)
        self.axis_y = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 20, 0]]), color='green', parent=self.view.scene)
        self.axis_z = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 0, 20]]), color='blue', parent=self.view.scene)

        #TODO if want to continue witht this visualizer: Add the body axis and rotate them per timestep according to the attitude 
        self.body_x = scene.visuals.Line(pos=np.array([self.quadcopter_position, self.quadcopter_position + [1, 0, 0]]), color='red', parent=self.view.scene)
        self.body_y = scene.visuals.Line(pos=np.array([self.quadcopter_position, self.quadcopter_position + [0, 1, 0]]), color='green', parent=self.view.scene)
        self.body_z = scene.visuals.Line(pos=np.array([self.quadcopter_position, self.quadcopter_position + [0, 0, 1]]), color='blue', parent=self.view.scene)

        # Create obstacle visuals
        self.obstacle_visuals = []
        for obstacle in self.obstacles:
            obstacle_visual = self.create_obstacle_visual(obstacle.position, obstacle.radius)
            self.obstacle_visuals.append(obstacle_visual)

        # Create quadcopter visual
        half_length = 0.5
        half_width = 0.5
        half_height = 0.25
        quadcopter_vertices = np.array([
            (-half_length, -half_width, -half_height),  # Front bottom left
            (half_length, -half_width, -half_height),   # Front bottom right
            (half_length, half_width, -half_height),    # Front top right
            (-half_length, half_width, -half_height),   # Front top left
            (-half_length, -half_width, half_height),   # Back bottom left
            (half_length, -half_width, half_height),    # Back bottom right
            (half_length, half_width, half_height),     # Back top right
            (-half_length, half_width, half_height)     # Back top left
        ])

        # Define the faces of the quadcopter
        quadcopter_faces = np.array([
            [0, 1, 2], [0, 2, 3],   # Front face
            [4, 5, 6], [4, 6, 7],   # Back face
            [0, 4, 7], [0, 7, 3],   # Left side face
            [1, 5, 6], [1, 6, 2],   # Right side face
            [0, 1, 5], [0, 5, 4],   # Bottom face
            [2, 3, 7], [2, 7, 6]    # Top face
        ])

        # Define colors for each vertex
        quadcopter_colors = np.array([
            
            [0.5, 0.5, 0.5], # Grey face
            [0.5, 0.5, 0.5],
            [0, 0, 1],  # Blue face Up
            [0, 0, 1],

            [0.5, 0.5, 0.5], # Grey face
            [0.5, 0.5, 0.5],
            [1, 0, 0], # Red face Along x axis ie front face
            [1, 0, 0],

            [0.5, 0.5, 0.5], # Grey face
            [0.5, 0.5, 0.5],
            [0, 1, 0],  # Green face Along y axis ie left face
            [0, 1, 0],
        ])

        # Create quadcopter visual
        self.quadcopter = scene.visuals.Mesh(vertices=quadcopter_vertices, faces=quadcopter_faces, parent=self.view.scene, shading='smooth', face_colors=quadcopter_colors)

        # Set camera position and aspect ratio
        self.view.camera = 'arcball'
        # self.view.camera.set_range((0, 100), (0, 100), (-100, 100))
        self.view.camera.set_range((-10, 10, 10), (0, 0, 0))
        self.view.camera.aspect = self.size[0] / self.size[1]
        
        self.update_quad_visual(self.quadcopter_position,self.quadcopter_attitude)

    def update_quad_visual(self, position, attitude):
        #TODO make attitude of quadcopter and body axis change correctly 

        #Translate the quadcopter mesh to the new position
        self.quadcopter.transform = scene.transforms.MatrixTransform()
        self.quadcopter.transform.translate(position)
        #Rotate the quadcopter mesh to the new attitude
        self.quadcopter.transform.rotate(np.degrees(attitude[0]), (1, 0, 0))  # Rotate around x-axis #might need to flip the sign
        self.quadcopter.transform.rotate(np.degrees(attitude[1]), (0, 1, 0))  # Rotate around y-axis
        self.quadcopter.transform.rotate(np.degrees(attitude[2]), (0, 0, 1))  # Rotate around z-axis

    def draw_path(self, waypoints):
        if self.waypoints_visual is not None:
            self.waypoints_visual.parent = None  # Remove previous waypoints if they exist

        self.waypoints = waypoints
        path = scene.visuals.Line(pos=np.array(waypoints), color='cyan', parent=self.view.scene)
        waypoints_markers = scene.visuals.Markers(parent=self.view.scene)
        waypoints_markers.set_data(pos=np.array(waypoints), size=10, edge_color='green', face_color='green')
        self.waypoints_visual = waypoints_markers

    def create_obstacle_visual(self, position, radius):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = position[0] + radius * np.cos(u) * np.sin(v)
        y = position[1] + radius * np.sin(u) * np.sin(v)
        z = position[2] + radius * np.cos(v)
        vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        faces = []
        for i in range(len(x) - 1):
            for j in range(len(x[0]) - 1):
                faces.append([i * len(x[0]) + j,
                              i * len(x[0]) + j + 1,
                              (i + 1) * len(x[0]) + j + 1])
                faces.append([i * len(x[0]) + j,
                              (i + 1) * len(x[0]) + j + 1,
                              (i + 1) * len(x[0]) + j])
        return scene.visuals.Mesh(vertices, faces, color='red', parent=self.view.scene)

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        self.update()
