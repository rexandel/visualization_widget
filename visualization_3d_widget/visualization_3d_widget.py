from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import QTimer, Qt

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np

class Visualization3DWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        glutInit()
        super().__init__(parent)

        self.default_rotation_x = -50
        self.default_rotation_y = 0
        self.default_rotation_z = 20
        self.default_zoom_level = 30
        self.default_position_x = 0.0
        self.default_position_y = 0.0

        self.rotation_x = self.default_rotation_x
        self.rotation_y = self.default_rotation_y
        self.rotation_z = self.default_rotation_z
        self.zoom_level = self.default_zoom_level
        self.position_x = self.default_position_x
        self.position_y = self.default_position_y

        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.is_rotating = False
        self.is_moving = False

        self.grid_size_x = 10
        self.grid_size_y = 10
        self.grid_size_z = 10
        self.grid_step = 1
        self.resolution = 250

        self.grid_visible = True
        self.axes_visible = True
        self.axis_ticks_and_numbers_visible = True

        self.current_function = None
        self.constraints = []
        self.show_constraints = False
        self.objective_function_data = None
        self.display_lists = {}
        self.z_min = 0
        self.z_max = 0
        self.optimization_path = np.array([])
        self.connect_optimization_points = True

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(16)

    def restore_default_view(self):
        self.rotation_x = self.default_rotation_x
        self.rotation_y = self.default_rotation_y
        self.rotation_z = self.default_rotation_z
        self.zoom_level = self.default_zoom_level
        self.position_x = self.default_position_x
        self.position_y = self.default_position_y
        self.update()

    def initializeGL(self):
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        light_position = [10.0, 10.0, 10.0, 1.0]
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, width, height):
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, self.zoom_level, 0, 0, 0, 0, 1, 0)
        glTranslatef(self.position_x, self.position_y, 0)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_POLYGON_SMOOTH)
        glDisable(GL_LINE_SMOOTH)

        if self.grid_visible:
            self.render_grid()

        if self.axes_visible:
            self.render_axes()

        if self.current_function and self.objective_function_data:
            if 'function' not in self.display_lists:
                self.display_lists['function'] = self.create_function_display_list()
            glCallList(self.display_lists['function'])

        self.draw_optimization_path()

        if self.show_constraints and self.constraints:
            self.draw_constraints()

    def create_function_display_list(self):
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)
        if self.objective_function_data:
            for strip in self.objective_function_data:
                glBegin(GL_QUAD_STRIP)
                current_strip_valid = False
                for idx, (vertex, color) in enumerate(strip):
                    current_valid = not np.isnan(vertex[2])
                    if current_valid:
                        glColor3f(*color)
                        glVertex3f(*vertex)
                        current_strip_valid = True
                    elif current_strip_valid:
                        glEnd()
                        glBegin(GL_QUAD_STRIP)
                        current_strip_valid = False
                glEnd()
        glEndList()
        return display_list

    def build_objective_function_data(self):
        if self.current_function is None:
            return
        x_values = np.linspace(-self.grid_size_x, self.grid_size_x, self.resolution)
        y_values = np.linspace(-self.grid_size_y, self.grid_size_y, self.resolution)
        z_values = np.full((len(x_values), len(y_values)), np.nan)
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                x = x_values[i]
                y = y_values[j]
                if self.constraints:
                    valid = all(constraint(x, y) <= 0 for constraint in self.constraints)
                else:
                    valid = True
                if valid:
                    z_values[i, j] = self.current_function(x, y)
        valid_z = z_values[~np.isnan(z_values)]
        if len(valid_z) > 0:
            self.z_min = np.min(valid_z)
            self.z_max = np.max(valid_z)
        else:
            self.z_min = 0
            self.z_max = 1
        shadow_strength = 0.6
        self.objective_function_data = []
        for i in range(len(x_values) - 1):
            strip = []
            for j in range(len(y_values)):
                x1 = x_values[i]
                x2 = x_values[i + 1]
                y = y_values[j]
                z1 = z_values[i, j]
                z2 = z_values[i + 1, j]
                z1_norm = (z1 - self.z_min) / (
                        self.z_max - self.z_min) * 2 * self.grid_size_z - self.grid_size_z if not np.isnan(
                    z1) else np.nan
                z2_norm = (z2 - self.z_min) / (
                        self.z_max - self.z_min) * 2 * self.grid_size_z - self.grid_size_z if not np.isnan(
                    z2) else np.nan
                if not np.isnan(z1):
                    z1_shadow = ((z1 - self.z_min) / (self.z_max - self.z_min)) ** 0.5
                    shadow_intensity1 = 1.0 - shadow_strength * (1.0 - z1_shadow)
                    color1 = (((x1 + self.grid_size_x) / (2 * self.grid_size_x)) * shadow_intensity1,
                              ((y + self.grid_size_y) / (2 * self.grid_size_y)) * shadow_intensity1,
                              0.7 * shadow_intensity1)
                else:
                    color1 = (0, 0, 0)
                if not np.isnan(z2):
                    z2_shadow = ((z2 - self.z_min) / (self.z_max - self.z_min)) ** 0.5
                    shadow_intensity2 = 1.0 - shadow_strength * (1.0 - z2_shadow)
                    color2 = (((x2 + self.grid_size_x) / (2 * self.grid_size_x)) * shadow_intensity2,
                              ((y + self.grid_size_y) / (2 * self.grid_size_y)) * shadow_intensity2,
                              0.7 * shadow_intensity2)
                else:
                    color2 = (0, 0, 0)
                strip.append(((x1, y, z1_norm), color1))
                strip.append(((x2, y, z2_norm), color2))
            self.objective_function_data.append(strip)
        if 'function' in self.display_lists:
            glDeleteLists(self.display_lists['function'], 1)
            self.display_lists.pop('function')

    # Work with optimization path

    def draw_optimization_path(self):
        if self.optimization_path.size == 0:
            return

        points = np.array(self.optimization_path, dtype=np.float32)
        z_values = np.zeros(len(points))

        for i in range(len(points)):
            z_values[i] = self.current_function(points[i, 0], points[i, 1])
        z_norm = (z_values - self.z_min) / (self.z_max - self.z_min) * 2 * self.grid_size_z - self.grid_size_z
        vertices = np.column_stack((points, z_norm)).astype(np.float32)

        glPointSize(10)
        glColor3f(1, 0, 0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glDrawArrays(GL_POINTS, 0, len(vertices))

        if self.connect_optimization_points:
            glLineWidth(2)
            glDrawArrays(GL_LINE_STRIP, 0, len(vertices))

        glDisableClientState(GL_VERTEX_ARRAY)

    def update_optimization_path(self, points):
        self.optimization_path = points
        self.update()

    ### Number rendering

    def render_number_0(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_1(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x, y + size / 2, z)
        glVertex3f(x, y - size / 2, z)
        glVertex3f(x - size / 4, y + size / 2, z)
        glVertex3f(x, y + size / 2, z)
        glEnd()

    def render_number_2(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glEnd()

    def render_number_3(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_4(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_5(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_6(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_7(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glEnd()

    def render_number_8(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_number_9(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y, z)
        glEnd()

    def render_number_minus(self, x, y, z, size):
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y, z)
        glVertex3f(x + size / 2, y, z)
        glEnd()

    def render_negative_number(self, x, y, z, number, size):
        self.render_number_minus(x - size, y, z, size / 2)
        self.render_number(x, y, z, abs(number), size)

    def render_number(self, x, y, z, number, size):
        glPushMatrix()
        glTranslatef(x, y, z)
        glPushMatrix()
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        glLoadIdentity()
        glTranslatef(modelview[3][0], modelview[3][1], modelview[3][2])
        scale = 0.5
        glScalef(scale, scale, scale)
        if number < 0:
            self.render_negative_number(0, 0, 0, number, size)
        else:
            digits = str(number)
            total_width = len(digits) * size * 0.75
            start_x = -total_width / 2 + size / 2
            for digit in digits:
                digit = int(digit)
                if digit == 0:
                    self.render_number_0(start_x, 0, 0, size)
                elif digit == 1:
                    self.render_number_1(start_x, 0, 0, size)
                elif digit == 2:
                    self.render_number_2(start_x, 0, 0, size)
                elif digit == 3:
                    self.render_number_3(start_x, 0, 0, size)
                elif digit == 4:
                    self.render_number_4(start_x, 0, 0, size)
                elif digit == 5:
                    self.render_number_5(start_x, 0, 0, size)
                elif digit == 6:
                    self.render_number_6(start_x, 0, 0, size)
                elif digit == 7:
                    self.render_number_7(start_x, 0, 0, size)
                elif digit == 8:
                    self.render_number_8(start_x, 0, 0, size)
                elif digit == 9:
                    self.render_number_9(start_x, 0, 0, size)
                start_x += size * 0.75
        glPopMatrix()
        glPopMatrix()

    ### Axis rendering

    def render_axis_label(self, x, y, z, label, color=(0.0, 0.0, 0.0)):
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glPushMatrix()
        glTranslatef(x, y, z)
        glPushMatrix()
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        glLoadIdentity()
        glTranslatef(modelview[3][0], modelview[3][1], modelview[3][2])
        scale = 0.5
        glScalef(scale, scale, scale)
        size = 0.5
        if label == "X":
            self.render_x_symbol(0, 0, 0, size)
        elif label == "Y":
            self.render_y_symbol(0, 0, 0, size)
        elif label == "Z":
            self.render_z_symbol(0, 0, 0, size)
        glPopMatrix()
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def render_x_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glEnd()

    def render_y_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y - size / 2, z)
        glEnd()

    def render_z_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_axes(self):
        if not self.axes_visible:
            return

        glDisable(GL_LIGHTING)

        glLineWidth(2)
        glBegin(GL_LINES)

        axis_extension = 0.2
        glColor3f(1, 0, 0)
        glVertex3f(-self.grid_size_x - axis_extension, 0, 0)
        glVertex3f(self.grid_size_x + axis_extension, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, -self.grid_size_y - axis_extension, 0)
        glVertex3f(0, self.grid_size_y + axis_extension, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -self.grid_size_z - axis_extension)
        glVertex3f(0, 0, self.grid_size_z + axis_extension)
        glEnd()

        if self.axis_ticks_and_numbers_visible:
            self.render_axis_ticks()

        offset = 0.3
        self.render_axis_label(self.grid_size_x + offset, 0, 0, "X", (0, 0, 0))
        self.render_axis_label(0, self.grid_size_y + offset, 0, "Y", (0, 0, 0))
        self.render_axis_label(0, 0, self.grid_size_z + offset, "Z", (0, 0, 0))
        glLineWidth(1)
        glEnable(GL_LIGHTING)

    def render_axis_ticks(self):
        if not self.axis_ticks_and_numbers_visible:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        tick_size = 0.2
        label_offset = 0.3
        label_size = 0.3

        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        for i in range(-self.grid_size_x, self.grid_size_x + 1, self.grid_step):
            if i != 0:
                glVertex3f(i, -tick_size / 2, 0)
                glVertex3f(i, tick_size / 2, 0)
                glVertex3f(i, 0, -tick_size / 2)
                glVertex3f(i, 0, tick_size / 2)
        glEnd()

        glColor3f(0, 0, 0)
        for i in range(-self.grid_size_x, self.grid_size_x + 1, self.grid_step):
            if i != 0:
                self.render_number(i, -label_offset, 0, i, label_size)

        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        for i in range(-self.grid_size_y, self.grid_size_y + 1, self.grid_step):
            if i != 0:
                glVertex3f(-tick_size / 2, i, 0)
                glVertex3f(tick_size / 2, i, 0)
                glVertex3f(0, i, -tick_size / 2)
                glVertex3f(0, i, tick_size / 2)
        glEnd()

        glColor3f(0, 0, 0)
        for i in range(-self.grid_size_y, self.grid_size_y + 1, self.grid_step):
            if i != 0:
                self.render_number(-label_offset, i, 0, i, label_size)

        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        for i in range(-self.grid_size_z, self.grid_size_z + 1, self.grid_step):
            if i != 0:
                glVertex3f(-tick_size / 2, 0, i)
                glVertex3f(tick_size / 2, 0, i)
                glVertex3f(0, -tick_size / 2, i)
                glVertex3f(0, tick_size / 2, i)
        glEnd()

        glColor3f(0, 0, 0)
        for i in range(-self.grid_size_z, self.grid_size_z + 1, self.grid_step):
            if i != 0:
                self.render_number(0, -label_offset, i, i, label_size)

        glEnable(GL_LIGHTING)

    def render_grid(self):
        glLineWidth(1)
        glColor3f(0.7, 0.7, 0.7)
        z_position = -self.grid_size_z
        for i in range(-self.grid_size_x, self.grid_size_x + 1, self.grid_step):
            glBegin(GL_LINES)
            glVertex3f(i, -self.grid_size_y, z_position)
            glVertex3f(i, self.grid_size_y, z_position)
            glEnd()
        for i in range(-self.grid_size_y, self.grid_size_y + 1, self.grid_step):
            glBegin(GL_LINES)
            glVertex3f(-self.grid_size_x, i, z_position)
            glVertex3f(self.grid_size_x, i, z_position)
            glEnd()

    # Work with constraints

    def draw_constraints(self):
        glDisable(GL_LIGHTING)
        glColor3f(1, 0, 0)
        glLineWidth(2)

        for constraint in self.constraints:
            self.draw_constraint_boundary(constraint)

        glEnable(GL_LIGHTING)

    def draw_constraint_boundary(self, constraint):
        x = np.linspace(-self.grid_size_x, self.grid_size_x, self.resolution)
        y = np.linspace(-self.grid_size_y, self.grid_size_y, self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = constraint(X[i, j], Y[i, j])
        glBegin(GL_LINES)

        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                edges = []
                if Z[i, j] * Z[i + 1, j] <= 0:
                    t = abs(Z[i, j]) / (abs(Z[i, j]) + abs(Z[i + 1, j])) if abs(Z[i, j]) + abs(Z[i + 1, j]) > 0 else 0.5
                    x0 = X[i, j] + t * (X[i + 1, j] - X[i, j])
                    y0 = Y[i, j]
                    edges.append((x0, y0))
                if Z[i + 1, j] * Z[i + 1, j + 1] <= 0:
                    t = abs(Z[i + 1, j]) / (abs(Z[i + 1, j]) + abs(Z[i + 1, j + 1])) if abs(Z[i + 1, j]) + abs(
                        Z[i + 1, j + 1]) > 0 else 0.5
                    x0 = X[i + 1, j]
                    y0 = Y[i + 1, j] + t * (Y[i + 1, j + 1] - Y[i + 1, j])
                    edges.append((x0, y0))
                if Z[i, j + 1] * Z[i + 1, j + 1] <= 0:
                    t = abs(Z[i, j + 1]) / (abs(Z[i, j + 1]) + abs(Z[i + 1, j + 1])) if abs(Z[i, j + 1]) + abs(
                        Z[i + 1, j + 1]) > 0 else 0.5
                    x0 = X[i, j + 1] + t * (X[i + 1, j + 1] - X[i, j + 1])
                    y0 = Y[i, j + 1]
                    edges.append((x0, y0))
                if Z[i, j] * Z[i, j + 1] <= 0:
                    t = abs(Z[i, j]) / (abs(Z[i, j]) + abs(Z[i, j + 1])) if abs(Z[i, j]) + abs(Z[i, j + 1]) > 0 else 0.5
                    x0 = X[i, j]
                    y0 = Y[i, j] + t * (Y[i, j + 1] - Y[i, j])
                    edges.append((x0, y0))
                if len(edges) >= 2:
                    for k in range(len(edges) - 1):
                        x0, y0 = edges[k]
                        x1, y1 = edges[k + 1]
                        if self.current_function:
                            z0 = self.current_function(x0, y0)
                            z1 = self.current_function(x1, y1)
                            z0_norm = (z0 - self.z_min) / (
                                        self.z_max - self.z_min) * 2 * self.grid_size_z - self.grid_size_z
                            z1_norm = (z1 - self.z_min) / (
                                        self.z_max - self.z_min) * 2 * self.grid_size_z - self.grid_size_z
                        else:
                            z0_norm = z1_norm = 0
                        glVertex3f(x0, y0, z0_norm)
                        glVertex3f(x1, y1, z1_norm)
        glEnd()

    def add_constraint(self, constraint_func):
        if constraint_func not in self.constraints:
            self.constraints.append(constraint_func)
            self.build_objective_function_data()
            self.update()

    def clear_constraints(self):
        self.constraints.clear()
        self.build_objective_function_data()
        self.update()

    # Overridden methods

    def mousePressEvent(self, event):
        self.mouse_last_x = event.x()
        self.mouse_last_y = event.y()

        if event.modifiers() & Qt.ControlModifier:
            self.is_moving = True
        else:
            self.is_rotating = True

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom_level -= delta
        self.zoom_level = max(5, min(self.zoom_level, 50))
        self.update()

    def mouseMoveEvent(self, event):
        dx, dy = event.x() - self.mouse_last_x, event.y() - self.mouse_last_y

        if self.is_rotating:
            if event.modifiers() & Qt.ShiftModifier:
                self.rotation_z += dx / 5
            else:
                self.rotation_x += dy / 5
                self.rotation_y += dx / 5
        elif self.is_moving:
            movement_speed = 0.001 * self.zoom_level
            self.position_x += dx * movement_speed
            self.position_y -= dy * movement_speed

        self.mouse_last_x, self.mouse_last_y = event.x(), event.y()
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_rotating = False
        self.is_moving = False

    # Setters & Getters

    def set_connect_optimization_points(self, connect):
        self.connect_optimization_points = connect
        self.update()

    def set_function(self, func):
        self.current_function = func
        self.build_objective_function_data()
        self.update()

    def set_show_constraints(self, show):
        self.show_constraints = show
        self.update()

    def set_axes_visible(self, show):
        self.axes_visible = show
        self.update()

    def get_axes_visible(self):
        return self.axes_visible

    def set_axis_ticks_and_numbers_visible(self, show):
        self.axis_ticks_and_numbers_visible = show
        self.update()

    def get_axis_ticks_and_numbers_visible(self):
        return self.axis_ticks_and_numbers_visible

    def set_grid_visible(self, show):
        self.grid_visible = show
        self.update()

    def get_grid_visible(self):
        return self.grid_visible

    def get_grid_size_x(self):
        return self.grid_size_x

    def get_grid_size_y(self):
        return self.grid_size_y

    def get_grid_size_z(self):
        return self.grid_size_z

    def set_grid_size_x(self, size_x):
        self.grid_size_x = size_x
        self.build_objective_function_data()
        self.update()

    def set_grid_size_y(self, size_y):
        self.grid_size_y = size_y
        self.build_objective_function_data()
        self.update()

    def set_grid_size_z(self, size_z):
        self.grid_size_z = size_z
        self.build_objective_function_data()
        self.update()

    def get_resolution(self):
        return self.resolution

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.build_objective_function_data()
        self.update()

    def get_x_axis_range(self):
        return [-self.grid_size_x, self.grid_size_x]

    def get_y_axis_range(self):
        return [-self.grid_size_y, self.grid_size_y]

    def get_z_axis_range(self):
        return [-self.grid_size_z, self.grid_size_z]

    def get_all_axes_ranges(self):
        return (
            [-self.grid_size_x, self.grid_size_x],
            [-self.grid_size_y, self.grid_size_y],
            [-self.grid_size_z, self.grid_size_z]
        )
