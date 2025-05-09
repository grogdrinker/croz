import torch
import numpy as np
from pyuul import VolumeMaker # the main PyUUL module

def score_voxel(voxels,all_subvolumes):
    voxels = voxels.to_dense()
    all_subvolumes = all_subvolumes.unsqueeze(0)
    all_subvolumes = all_subvolumes * voxels
    loss = (all_subvolumes).mean(dim=-1).mean(dim=-1).mean(dim=-1)

    vals,max_subvolume = loss.max(dim=0)
    score,max_rotation = vals.max(dim=0)
    return score,max_subvolume[max_rotation],max_rotation

def get_initial_alignment(coords,radius,channels,electrondensity,size_of_a_voxel,random_tests=100000,downsampled_factor = 0.10):
    test_per_loop = 30
    todo_steps  = random_tests // test_per_loop + 1


    electrondensity = downsample_volume(electrondensity, scale_factor=downsampled_factor)
    size_of_a_voxel_donwsampled = size_of_a_voxel/downsampled_factor

    VoxelsObject = VolumeMaker.Voxels(device=coords.device, sparse=True)

    best_score = 0
    for i in range(todo_steps):
        print("start",i,best_score)
        with torch.no_grad():

            coordsTMP= coords.clone()/size_of_a_voxel_donwsampled

            coordsTMP=(coordsTMP-coordsTMP.max(dim=1)[0])[0]
            radiusTMP = radius[0].clone()
            radiusTMP /= size_of_a_voxel_donwsampled

            rotations = generate_transformations(test_per_loop,device=coords.device)


            transformed_coords = apply_rotations(coordsTMP, rotations)
            with torch.no_grad():
                N = transformed_coords.shape[0]

                voxels = VoxelsObject(transformed_coords, radius.expand(N,-1), channels.expand(N,-1))
                threshold = voxels[0].sum()*0.1
                all_subvolumes,mask = extract_subvolumes(electrondensity, voxels.shape[2], voxels.shape[3], voxels.shape[4],threshold=threshold)
                best_scoreTMP,maxRot,maxSubvol = score_voxel(voxels, all_subvolumes)

            if best_scoreTMP > best_score:
                best_score = best_scoreTMP
                translation = get_translation_from_subvolume(electrondensity.shape[0],voxels.shape[2], voxels.shape[3], voxels.shape[4],maxSubvol,mask)
                rotation = rotations[maxRot]

                #translation = torch.zeros(3,device=coords.device).float()#get_translation_from_subvolume(electrondensity.shape[0],voxels.shape[2], voxels.shape[3], voxels.shape[4],maxSubvol,mask)
                #rotation = torch.zeros(3,device=coords.device).float()

                # putting everything back to original reference system
                upsampled_coords = coordsTMP * size_of_a_voxel_donwsampled
                translation = translation * size_of_a_voxel_donwsampled

                final_coords = apply_rotations(upsampled_coords, rotation.unsqueeze(0)) + translation
    print("best score", best_score)
    final_coords/=size_of_a_voxel
    radius/=size_of_a_voxel
    return final_coords,radius


def generate_transformations(N,device="cpu"):
    rotations = (torch.rand(N, 3) * 2*torch.pi).to(device)  # Random rotation angles in radians
    return rotations

def apply_rotations(coords, rotations):
    """
    Applies N transformations (rotation around the center of mass + translation) to a set of M 3D coordinates.

    Args:
        coords (torch.Tensor): Input coordinates of shape (M, 3).
        rotations (torch.Tensor): Rotation vectors (Euler angles) of shape (N, 3).

    Returns:
        transformed_coords (torch.Tensor): Transformed coordinates of shape (N, M, 3).
    """
    N, M = rotations.shape[0], coords.shape[0]

    # Compute center of mass
    center_of_mass = coords.mean(dim=0, keepdim=True)
    centered_coords = coords - center_of_mass  # Centered coordinates

    # Convert Euler angles to rotation matrices using Rodrigues' formula
    def rodrigues_rotation(angles):
        theta = torch.norm(angles, dim=1, keepdim=True)  # Rotation magnitudes
        unit_vecs = angles / (theta + 1e-8)  # Normalize (avoid div by zero)

        K = torch.zeros(N, 3, 3).to(angles.device)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -unit_vecs[:, 2], unit_vecs[:, 1], unit_vecs[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -unit_vecs[:, 0], -unit_vecs[:, 1], unit_vecs[:, 0]

        I = torch.eye(3).to(angles.device).expand(N, 3, 3)
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * (K @ K)
        return R

    R = rodrigues_rotation(rotations)  # Shape (N, 3, 3)

    # Apply rotation around the center of mass
    coords_expanded = centered_coords.unsqueeze(0).expand(N, M, 3)  # Shape (N, M, 3)
    rotated_coords = torch.einsum('nij,nmj->nmi', R, coords_expanded)  # (N, M, 3)
    rotated_coords += center_of_mass.unsqueeze(0)  # Translate back to original position

    # Apply translation
    transformed_coords = rotated_coords

    return transformed_coords


def sample_rotations(N,points=10000):
    """
    Generates N evenly spaced 3D rotations using a structured approach.

    Args:
        N (int): Number of rotations.

    Returns:
        torch.Tensor: Euler angles (N, 3) representing sampled rotations in radians.
    """
    # Sample yaw and pitch from Fibonacci sphere
    def fibonacci_grid_on_sphere(points):
        """
        Generates N evenly distributed points on a sphere using the Fibonacci lattice.

        Args:
            N (int): Number of points.

        Returns:
            torch.Tensor: Spherical coordinates (theta, phi) for each point (N, 2).
        """
        indices = torch.arange(N, dtype=torch.float)
        phi = torch.acos(1 - 2 * (indices + 0.5) / N)  # Latitude (0 to pi)
        theta = torch.pi * (1 + 5 ** 0.5) * indices  # Longitude (0 to 2pi)
        return torch.stack([theta % (2 * torch.pi), phi], dim=1)
    spherical_coords = fibonacci_grid_on_sphere(N)
    spherical_coords = spherical_coords[torch.randint(high=spherical_coords.shape[0], size=(N,))]
    yaw, pitch = spherical_coords[:, 0], spherical_coords[:, 1]

    # Roll angle should also be evenly spaced
    roll = torch.rand(N) * 2 * torch.pi

    # Combine into (yaw, pitch, roll)
    return torch.stack([yaw, pitch, roll], dim=1)


def extract_subvolumes(volume, N, O, P, threshold=0.1):
    """
    Extracts all subvolumes of size (N, O, P) from a 3D tensor (M, M, M),
    ignoring subvolumes where the sum of values is close to zero.

    Args:
        volume (torch.Tensor): Input tensor of shape (M, M, M).
        N (int): Subvolume size along the first dimension.
        O (int): Subvolume size along the second dimension.
        P (int): Subvolume size along the third dimension.
        threshold (float): Minimum sum required for a subvolume to be included.

    Returns:
        torch.Tensor: Extracted subvolumes of shape (K, N, O, P), where K is the number of valid subvolumes.
    """
    #volume= volume.type(torch.bfloat16)
    M = volume.shape[0]

    # Ensure the subvolume size is not larger than the input volume
    assert N <= M and O <= M and P <= M, "Subvolume size must be <= volume size"

    # Extract all subvolumes using F.unfold equivalent for 3D
    subvolumes = volume.unfold(0, N, 1).unfold(1, O, 1).unfold(2, P, 1)  # Shape: (M-N+1, M-O+1, M-P+1, N, O, P)

    sums = subvolumes.sum(dim=(-1, -2, -3))  # Shape: (K,)
    valid_mask = sums > threshold

    # Reshape into a list of subvolumes
    subvolumes = subvolumes[valid_mask]#.contiguous().view(-1, N, O, P)  # Shape: (K, N, O, P), where K is number of subvolumes

    # Compute sum for each subvolume and filter

    #valid_mask = sums.abs() > threshold  # Mask out near-zero subvolumes

    return subvolumes,valid_mask


def clamp_volume(volume, threshold=0.001):
    """Clamps a 3D volume by removing the outer empty regions below a threshold."""
    mask = volume > threshold  # Boolean mask of non-empty values
    nonzero_indices = torch.nonzero(mask)  # Get indices of non-empty voxels

    if nonzero_indices.numel() == 0:  # If all values are below threshold, return an empty volume
        return volume[:0, :0, :0]

    min_coords, _ = nonzero_indices.min(dim=0)
    max_coords, _ = nonzero_indices.max(dim=0)

    return volume[min_coords[0]:max_coords[0] + 1,
           min_coords[1]:max_coords[1] + 1,
           min_coords[2]:max_coords[2] + 1]


def downsample_volume(volume, scale_factor=0.5):
    """
    Downsamples a 3D volume by a given scale factor.

    Args:
        volume (torch.Tensor): Input volume of shape (M, M, M).
        scale_factor (float): Factor by which to reduce resolution (0 < scale_factor ≤ 1).

    Returns:
        torch.Tensor: Downsampled volume.
    """
    assert 0 < scale_factor <= 1, "Scale factor must be between 0 and 1."

    M = volume.shape[0]
    new_size = int(M * scale_factor)

    # Reshape to (1, 1, M, M, M) for 3D interpolation
    volume = volume.unsqueeze(0).unsqueeze(0)

    # Use trilinear interpolation to resize
    downsampled = torch.nn.functional.interpolate(volume, size=(new_size, new_size, new_size), mode="trilinear", align_corners=False)

    return downsampled.squeeze(0).squeeze(0)  # Remove batch and channel dimensions


def get_translation_from_subvolume(M, N, O, P, subvolume_index,valid_mask):
    """
    Given an index of the extracted subvolumes, return the translation vector (in voxel units)
    corresponding to its position in the original volume.

    Args:
        volume (torch.Tensor): Input tensor of shape (M, M, M).
        N (int): Subvolume size along the first dimension.
        O (int): Subvolume size along the second dimension.
        P (int): Subvolume size along the third dimension.
        valid_mask (torch.Tensor): Boolean mask indicating which subvolumes are valid.
        subvolume_index (int): The index (after filtering) of the subvolume.

    Returns:
        torch.Tensor: Translation vector [x, y, z] (voxel units) in the original volume.
    """

    # Compute grid of all subvolume starting positions before masking
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(M - N + 1),
        torch.arange(M - O + 1),
        torch.arange(M - P + 1),
        indexing="ij"
    )

    # Flatten to match the valid_mask shape
    all_positions = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1).to(valid_mask.device)  # Shape: (K, 3)

    # Apply valid mask to get only the positions of kept subvolumes
    valid_positions = all_positions[valid_mask.flatten()]  # Shape: (K', 3)

    # Return the translation vector of the subvolume at subvolume_index
    return valid_positions[subvolume_index].float()  # Shape: (3,)