from setuptools import setup, find_packages

setup(
    name='Visualization3DWidgetPlugin',
    version='0.1',
    packages=find_packages(),
    package_data={
        'visualization_3d_widget': ['*.ui', '*.py', '__init__.py']
    },
    entry_points={
        'pyqt5_designer_plugins': [
            'Visualization3DWidgetPlugin = visualization_3d_widget.plugin:Visualization3DWidgetPlugin'
        ]
    },
    install_requires=[
        'PyQt5>=5.15',
        'numpy>=1.20',
        'PyOpenGL>=3.1.5'
    ],
    python_requires='>=3.6',
)