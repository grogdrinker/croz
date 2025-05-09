from pyuul import VolumeMaker
PADDING_INDEX = 999.0
import torch# the main PyUUL module
import math
import numpy as np
class vm(VolumeMaker.Voxels):
    def forward(self, coords, radius, channels,boxsize, numberchannels=None, resolution=1, cubes_around_atoms_dim=5,steepness=1, function="sigmoid"):
        """
        Voxels representation of the macromolecules

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of the atoms. Shape ( batch, numberOfAtoms, 3 ). Can be calculated from a PDB file using utils.parsePDB
        radius : torch.Tensor
            Radius of the atoms. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToRadius
        channels: torch.LongTensor
            channels of the atoms. Atoms of the same type shold belong to the same channel. Shape ( batch, numberOfAtoms ). Can be calculated from a PDB file using utils.parsePDB and utils.atomlistToChannels
        numberchannels : int or None
            maximum number of channels. if None, max(atNameHashing) + 1 is used

        cubes_around_atoms_dim : int
            maximum distance in number of voxels for which the contribution to occupancy is taken into consideration. Every atom that is farer than cubes_around_atoms_dim voxels from the center of a voxel does no give any contribution to the relative voxel occupancy
        resolution : float
            side in A of a voxel. The lower this value is the higher the resolution of the final representation will be

        steepness : float or int
            steepness of the sigmoid occupancy function.

        function : "sigmoid" or "gaussian"
            occupancy function to use. Can be sigmoid (every atom has a sigmoid shaped occupancy function) or gaussian (based on Li et al. 2014)
        Returns
        -------
        volume : torch.Tensor
            voxel representation of the macromolecules in the batch. Shape ( batch, channels, x,y,z), where x,y,z are the size of the 3D volume in which the macromolecules have been represented

        """
        padding_mask = ~channels.eq(PADDING_INDEX)
        if numberchannels is None:
            numberchannels = int(channels[padding_mask].max().cpu().data + 1)
        self.featureVectorSize = numberchannels
        self.function = function

        arange_type = torch.int16

        gx = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=self.dev, dtype=arange_type)
        gy = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=self.dev, dtype=arange_type)
        gz = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=self.dev, dtype=arange_type)
        self.lato = gx.shape[0]

        x1 = gx.unsqueeze(1).expand(self.lato, self.lato).unsqueeze(-1)
        x2 = gy.unsqueeze(0).expand(self.lato, self.lato).unsqueeze(-1)

        xy = torch.cat([x1, x2], dim=-1).unsqueeze(2).expand(self.lato, self.lato, self.lato, 2)
        x3 = gz.unsqueeze(0).unsqueeze(1).expand(self.lato, self.lato, self.lato).unsqueeze(-1)

        del gx, gy, gz, x1, x2

        self.standard_cube = torch.cat([xy, x3], dim=-1).unsqueeze(0).unsqueeze(0)

        ### definition of the box ###
        # you take the maximum and min coord on each dimension (every prot in the batch shares the same box. In the future we can pack, but I think this is not the bottleneck)
        # I scale by resolution
        # I add the cubes in which I define the gradient. One in the beginning and one at the end --> 2*



        #############################

        self.translation = 0.0
        self.dilatation = 1.0 / resolution

        boxsize = (int(boxsize[0]), int(boxsize[1]), int(boxsize[2]))
        self.boxsize = boxsize

        # selecting best types for indexing
        if max(boxsize) < 256:  # i can use byte tensor
            self.dtype_indices = torch.uint8
        else:
            self.dtype_indices = torch.int16

        if self.function == "sigmoid":
            volume = self._Voxels__forward_actual_calculation(coords+1.5, boxsize, radius, channels, padding_mask, steepness,
                                                       resolution)
        elif self.function == "gaussian":
            volume = self._Voxels__forward_actual_calculationGaussian(coords+1.5, boxsize, radius, channels, padding_mask,
                                                               resolution)
        return volume

def score_cloudpoint(cp,electrondensity):

    ind = cp.long()
    return electrondensity[ind[..., 0], ind[..., 1], ind[..., 2]].mean(dim=1)

def get_initial_alignment(coords,radius,electrondensity,size_of_a_voxel,maxpoints=2000,random_tests=50000):
    test_per_loop = 30000
    todo_steps  = random_tests // test_per_loop + 1
    best_score = 0.0

    def test_crio(img2, coords):
        map = torch.zeros(img2.shape)
        c = coords.cpu().data.tolist()
        for k in c:
            x = round(float(k[0]))
            y = round(float(k[1]))
            z = round(float(k[2]))
            map[x, y, z] = 1.0
        return np.array(map)

    for i in range(todo_steps):
        with torch.no_grad():
            PointCloudVolumeObject = VolumeMaker.PointCloudVolume(device=coords.device)
            coordsTMP= coords.clone()*size_of_a_voxel

            coordsTMP=(coordsTMP-coordsTMP.max(dim=1)[0])[0]
            radiusTMP = radius[0].clone()
            radiusTMP*=size_of_a_voxel

            translations, rotations = generate_transformations(test_per_loop, list(electrondensity.shape),device=coords.device)

            transformed_coords = apply_transformations(coordsTMP, translations, rotations)

            #cp = PointCloudVolumeObject(transformed_coords, radiusTMP.unsqueeze(0).expand(test_per_loop,-1),maxpoints=maxpoints)
            cp = transformed_coords
            mask = ((torch.tensor(electrondensity.shape,device=coords.device,dtype=torch.long) - cp.max(dim=1)[0])>0).prod(-1).bool() & cp.min(dim=1)[0].prod(-1).bool() > 0

            vals = score_cloudpoint(cp[mask],electrondensity)
            best_scoreTMP,best = vals.max(dim=0)

            if best_scoreTMP > best_score:
                best_score = best_scoreTMP
                new_coords_unscaled = transformed_coords[mask][best]
                new_coords = apply_transformations(coordsTMP / size_of_a_voxel,
                                                   translations[mask][best].unsqueeze(0) / size_of_a_voxel,
                                                   rotations[mask][best].unsqueeze(0))
                try:
                    assert new_coords_unscaled.min() > 0 and new_coords_unscaled.max() < electrondensity.shape[0]
                except:
                    asd
                best_new_coords = new_coords
    print("best score", best_score)
    """

    tuo_vol = test_crio(electrondensity,new_coords_unscaled)

    from VoxelPlotter import plot_3d_voxels, init
    ax_main = init()

    # gs = gridspec.GridSpec(5, 3, height_ratios=[1] * 5, width_ratios=[1, 3, 1])
    # ax_main = plt.subplot(gs[:, 1], projection='3d')
    import matplotlib.pylab as plt
    tresh = torch.nn.Threshold(0.039, 0.0)
    plot_3d_voxels(tuo_vol, ax_main, color="b")
    plot_3d_voxels(tresh(electrondensity).cpu().data.numpy(), ax_main, color="r")
    plt.show()
    asdasdasd
    """



    return new_coords_unscaled.unsqueeze(0),radiusTMP.unsqueeze(0)


def kabsch_algorithm(source: torch.Tensor, target: torch.Tensor):
    assert source.shape == target.shape, "Source and target must have the same shape"

    # Compute centroids
    centroid_source = source.mean(dim=0)
    centroid_target = target.mean(dim=0)

    # Center the points
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Compute covariance matrix
    H = source_centered.T @ target_centered

    # Singular Value Decomposition
    U, S, Vt = torch.svd(H)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a right-handed coordinate system (avoid reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Compute translation vector
    t = centroid_target - R @ centroid_source

    return R, t


def generate_transformations(N,MaxTranslation,device="cpu"):
    """
    Generates N random 3D translation and rotation vectors.

    Args:
        N (int): Number of transformations.

    Returns:
        translations (torch.Tensor): Tensor of shape (N, 3), random translations.
        rotations (torch.Tensor): Tensor of shape (N, 3), random rotation angles (radians).
    """
    translations = torch.rand(N, 3).to(device)
    translations = translations*torch.tensor(MaxTranslation).type_as(translations)# Random translations
    rotations = (torch.rand(N, 3) * 2*torch.pi).type_as(translations)  # Random rotation angles in radians
    return translations, rotations


def apply_transformations(coords, translations, rotations):
    """
    Applies N transformations (rotation around the center of mass + translation) to a set of M 3D coordinates.

    Args:
        coords (torch.Tensor): Input coordinates of shape (M, 3).
        translations (torch.Tensor): Translation vectors of shape (N, 3).
        rotations (torch.Tensor): Rotation vectors (Euler angles) of shape (N, 3).

    Returns:
        transformed_coords (torch.Tensor): Transformed coordinates of shape (N, M, 3).
    """
    N, M = translations.shape[0], coords.shape[0]

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
    transformed_coords = rotated_coords + translations.unsqueeze(1)  # (N, M, 3)

    return transformed_coords


