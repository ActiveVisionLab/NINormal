import torch
import math
import numpy as np


def comp_ang(pred_n, gt_n):
    """
    :param pred_n:  (N, 3)
    :param gt_n:    (N, 3)
    :return:        a scalar, average angle between predicted normals and gt normals
    """
    # for un-orient normal vector, it's fine if it's flipped, cos(theta) = -1 means correct normal.
    # clamp() because sometime cosine_similarity generate value slightly larger than 1.
    cos_dists = torch.abs(torch.nn.functional.cosine_similarity(pred_n, gt_n, dim=1, eps=1e-8)).clamp(-1.0, +1.0)
    angles_rad = torch.acos(cos_dists)
    angles_deg = 180 * angles_rad / math.pi
    avg_angles = torch.mean(angles_deg)
    return avg_angles


def comp_ang_batch(pred_n, gt_n):
    """
    :param pred_n:  (B, N, 3)
    :param gt_n:    (B, N, 3)
    :return:        a scalar, average angle between predicted normals and gt normals
    """
    # for un-orient normal vector, it's fine if it's flipped, cos(theta) = -1 means correct normal.
    # clamp() because sometime cosine_similarity generate value slightly larger than 1.
    cos_dists = torch.abs(torch.nn.functional.cosine_similarity(pred_n, gt_n, dim=2, eps=1e-8)).clamp(-1.0, +1.0)
    angles_rad = torch.acos(cos_dists)
    angles_deg = 180 * angles_rad / float(math.pi)
    avg_angles = torch.mean(angles_deg)  # this has already divided by B and N

    return avg_angles


def comp_pgp(pred_n, gt_n):
    """
    :param pred_n:  (N, 3)
    :param gt_n:    (N, 3)
    :return:        number of normals within certain degree thresholds
    """
    cos_dists = torch.abs(torch.nn.functional.cosine_similarity(pred_n, gt_n, dim=1, eps=1e-8)).clamp(-1.0, +1.0)
    angles_rad = torch.acos(cos_dists)  # always produce angle between (0, pi)
    angles_deg = 180 * angles_rad / math.pi
    angles_deg = torch.where(angles_deg > 90, 180 - angles_deg, angles_deg)  # convert 150 degree -> 30 degree

    pgp003 = torch.sum(angles_deg <= 3)
    pgp005 = torch.sum(angles_deg <= 5)
    pgp010 = torch.sum(angles_deg <= 10)
    pgp030 = torch.sum(angles_deg <= 30)
    pgp060 = torch.sum(angles_deg <= 60)
    pgp090 = torch.sum(angles_deg <= 90)

    return {
        'pgp003': pgp003,
        'pgp005': pgp005,
        'pgp010': pgp010,
        'pgp030': pgp030,
        'pgp060': pgp060,
        'pgp090': pgp090,
    }


def comp_pgp_batch_unori(pred_n, gt_n):
    """
    :param pred_n: (B, N, 3)
    :param gt_n:   (B, N, 3)
    :return:
    """
    cos_dists = torch.abs(torch.nn.functional.cosine_similarity(pred_n, gt_n, dim=2, eps=1e-8)).clamp(-1.0, +1.0)
    angles_rad = torch.acos(cos_dists)  # always produce angle between (0, pi)
    angles_deg = 180 * angles_rad / math.pi
    angles_deg = torch.where(angles_deg > 90, 180 - angles_deg, angles_deg)  # convert 150 degree -> 30 degree

    pgp003 = torch.sum(angles_deg <= 3)
    pgp005 = torch.sum(angles_deg <= 5)
    pgp010 = torch.sum(angles_deg <= 10)
    pgp030 = torch.sum(angles_deg <= 30)
    pgp060 = torch.sum(angles_deg <= 60)
    pgp080 = torch.sum(angles_deg <= 80)
    pgp090 = torch.sum(angles_deg <= 90)

    return {
        'pgp003': pgp003,
        'pgp005': pgp005,
        'pgp010': pgp010,
        'pgp030': pgp030,
        'pgp060': pgp060,
        'pgp080': pgp080,
        'pgp090': pgp090,
    }


def comp_rms_angle_batch(pred_n, gt_n):
    """
    :param pred_n: (B, N, 3)
    :param gt_n:   (B, N, 3)
    :return:
    """
    cos_dists = torch.abs(torch.nn.functional.cosine_similarity(pred_n, gt_n, dim=2, eps=1e-8)).clamp(-1.0, +1.0)
    angles_rad = torch.acos(cos_dists)  # always produce angle between (0, pi)
    angles_deg = 180 * angles_rad / math.pi
    angles_deg = torch.where(angles_deg > 90, 180 - angles_deg, angles_deg)  # convert 150 degree -> 30 degree

    rms_angle_error = torch.sqrt(angles_deg.pow(2).sum() / (angles_deg.view(-1).shape[0]))
    return rms_angle_error.item()
