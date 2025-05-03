from setuptools import setup

setup(
    name='Visualization3DWidgetPlugin',
    version='0.1',
    packages=['visualization_3d_widget'],
    package_data={'visualization_3d_widget': ['*.ui']},
    entry_points={
        'pyqt5_designer_plugins': [
            'Visualization3DWidgetPlugin = visualization_3d_widget.plugin:Visualization3DWidgetPlugin'
        ]
    },
)