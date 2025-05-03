from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from PyQt5.QtGui import QIcon

class Visualization3DWidgetPlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialized = False

    def initialize(self, core):
        if self.initialized:
            return
        self.initialized = True

    def isInitialized(self):
        return self.initialized

    def createWidget(self, parent):
        return Visualization3DWidget(parent)

    def name(self):
        return "Visualization3DWidget"

    def group(self):
        return "3D Visualization Widgets"

    def icon(self):
        return QIcon()

    def toolTip(self):
        return "3D Visualization Widget based on OpenGL"

    def whatsThis(self):
        return "This is a custom 3D visualization widget based on OpenGL."

    def isContainer(self):
        return False

    def includeFile(self):
        return "visualization_3d_widget"