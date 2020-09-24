import open3d as o3d
import numpy as np


def draw_object(pts):
    """
    :param pts:         (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


def draw_object_and_color(pts, colors):
    """
    :param pts:         (N, 3)
    :param colors:      (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def draw_object_and_normal(pts, normals):
    """
    :param pts:         (N, 3)
    :param normals:     (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


def draw_object_and_color_and_normal(pts, colors, normals):
    """
    :param pts:         (N, 3)
    :param colors:      (N, 3)
    :param normals:     (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


def draw_two_object(pts0, pts1, offset=np.array([2.5, 0, 0])):
    """
    :param pts0/1:         (N, 3)
    :param colors0/1:      (N, 3)
    :param offset:         (1, 3) display the second point cloud by the side. Default to the right.
    """
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1 + offset)
    o3d.visualization.draw_geometries([pcd0, pcd1])


def draw_two_object_and_color(pts0, pts1, colors0, colors1, offset=np.array([2.5, 0, 0])):
    """
    :param pts0/1:         (N, 3)
    :param colors0/1:      (N, 3)
    :param offset:         (1, 3) display the second point cloud by the side. Default to the right.
    """
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0)
    pcd0.colors = o3d.utility.Vector3dVector(colors0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1 + offset)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    o3d.visualization.draw_geometries([pcd0, pcd1])