import mrcfile

device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils as pyuulUtils# the PyUUL utility module

from croz.src.pyuulPatch import vm
from croz.src.initial_conformation import get_initial_alignment

VoxelsObject = vm(device=device, sparse=False)
function="gaussian"

from croz.src.utils import get_rotation_matrix,min_max_scale,writePDB


import torch,math

class LossPearsons:
	def __init__(self):
		self.mse=torch.nn.MSELoss()
		pass
	def __call__(self,pr,tr):
		pr=pr.view(-1)
		tr=tr.view(-1)
		#return self.mse(pr,tr)
		vx = pr - torch.mean(pr)
		vy = tr - torch.mean(tr)

		cost = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)+0.0000001) * torch.sqrt(torch.sum(vy ** 2)+0.0000001)))

		return -cost

def optimizeCryoEM(coords,atoms_channel,radius, img2,size_side_voxel,num_steps=1000, lr=0.0001,atom_names=None,out_pdb=False,verbose=False,rotate_side_chains=False):
    img2 = min_max_scale(img2)

    # Initial translation parameters
    translation = torch.zeros(3, requires_grad=True, device=device)
    rotation = torch.zeros(3, requires_grad=True, device=device)

    if rotate_side_chains:
        from madrax.mutate import rotator
        from madrax import  dataStructures
        from madrax.sources import hashings

        rotator_obj = rotator.RotateStruct()

        atnames = [[i+"_0_0" for i in atom_names[0]]]
        info_tensors = dataStructures.create_info_tensors(atnames, device=device)

        maxchain = info_tensors[1][:, hashings.atom_description_hash["chain"]].max() + 1
        maxseq = info_tensors[1][:, hashings.atom_description_hash["resnum"]].max() + 1

        torsion_rotation = torch.zeros((1, maxchain, maxseq, 1, 8), device=device, dtype=torch.float,requires_grad=True)
        #optimizer = torch.optim.Adam([torsion_rotation], lr=lr)

        optimizer = torch.optim.Adam([
            {'params': torsion_rotation, "lr": 0.1},
            {'params': translation, 'lr': lr},
            {'params': rotation, 'lr': lr}
        ], amsgrad=True, eps=0.1)

    else:
        optimizer = torch.optim.Adam([translation,rotation], lr=lr)

    radius = radius*1.4

    initial_conformation = False
    if initial_conformation:
        coords, radius = get_initial_alignment(coords,radius,atoms_channel,img2, size_side_voxel)
    else:
        coords = coords/size_side_voxel

    loss_fn = LossPearsons()
    best_score = 0.0
    for step in range(num_steps):

        optimizer.zero_grad()
        center_of_mass = coords.mean(dim=1)
        R = get_rotation_matrix(rotation[0], rotation[1], rotation[2])

        rotated_points = ((coords-center_of_mass).squeeze(0) @ R.T + center_of_mass + translation).unsqueeze(0)

        if rotate_side_chains:
            torsion_angles = torch.sin(torsion_rotation) * math.pi
            final_rot = rotator_obj(rotated_points, info_tensors, torsion_angles, backbone_rotation=False)
        else:
            final_rot = rotated_points

        rotated_volume = get_pyuul_voxels_new(list(img2.shape),atoms_channel=atoms_channel,radius=radius,coords=final_rot,pdb_fil = None)
        rotated_volume = min_max_scale(rotated_volume)

        mask = rotated_volume>0.001
        #loss = -torch.mean(rotated_volume[mask] * img2[mask]) #sinkhorn(rotated_points, img2)[0]

        loss = loss_fn(rotated_volume[mask].view(-1) , img2[mask].view(-1)) #sinkhorn(rotated_points, img2)[0]
        if float(loss.cpu().data)<best_score:
            best_points = rotated_points
            best_score = float(loss.cpu().data)
            best_rotation = rotation.cpu().data.tolist()
            best_translation = translation.cpu().data.tolist()
        if not rotate_side_chains and verbose:
            if step % 50 == 0:
                print("\tloss step",step,round(float(loss.data),5),"rotations",round(float(rotation.cpu().abs().mean().data),5),"translation",round(float(translation.cpu().abs().mean().data),5))
        else:
            if step % 50 == 0 and verbose:
                print("\tloss step",step,round(float(loss.data),5),"rotations",round(float(rotation.cpu().abs().mean().data),5),"translation",round(float(translation.cpu().abs().mean().data),5),"torsion_rotation",round(float(torsion_rotation.cpu().abs().mean().data),5))
        loss.backward()
        optimizer.step()
    if best_score == 0.0:
        print("OPTIMIZATION FAILED. I cannot fit the PDB model in the electron density")
        return None,None
    best_points*=size_side_voxel
    rotated_points = best_points
    best_score = -best_score
    if verbose:
        print("best score",best_score)
    if out_pdb:
        writePDB(rotated_points,atom_names,out_pdb)

    return best_score,best_points

def get_pyuul_voxels(shapes,size_side_voxel=1,atoms_channel=None,radius=None,coords=None,pdb_fil = None):
    size_side_voxel = 1
    cubes_around_atoms_dim = 5

    if atoms_channel is None:
        coords, atname = pyuulUtils.parsePDB(pdb_fil)  # get coordinates and atom names
        atoms_channel = torch.zeros((len(atname),len(atname[0])))
        radius = pyuulUtils.atomlistToRadius(atname)  # calculates the radius of each atom
        coords = coords.to(device)
        radius = radius.to(device)
        atoms_channel = atoms_channel.to(device)


    shapes[0] = (shapes[0]-2.5) * size_side_voxel
    shapes[1] = (shapes[1]-2.5) * size_side_voxel
    shapes[2] = (shapes[2]-2.5) * size_side_voxel

    fake_coords = torch.tensor([[0.0, 0.0,0.0],
                                [shapes[0], 0.0, 0.0],
                                [shapes[0], shapes[1], 0.0],
                                [shapes[0], shapes[1], shapes[2]],
                                [0.0, shapes[1], 0.0],
                                [0.0,  shapes[1],  shapes[2]],
                                [0.0, 0.0,  shapes[2]],
                                [shapes[0], 0.0,  shapes[2]],
                               ]).type_as(coords)



    pad_coords = torch.nn.utils.rnn.pad_sequence([coords[0],fake_coords],batch_first=True,padding_value=999.0)
    pad_radius = torch.nn.utils.rnn.pad_sequence([radius[0],torch.tensor([2]*fake_coords.shape[0]).type_as(radius)],batch_first=True,padding_value=999.0)
    atoms_channel_pad = torch.nn.utils.rnn.pad_sequence([atoms_channel[0], torch.tensor([0.0]*len(fake_coords)+[999.0] * (atoms_channel.shape[1]-len(fake_coords))).type_as(atoms_channel)],batch_first=True, padding_value=999.0)

    VoxelRepresentation = VoxelsObject(pad_coords, pad_radius, atoms_channel_pad,resolution = size_side_voxel,cubes_around_atoms_dim=cubes_around_atoms_dim,function=function)[0,0]

    return VoxelRepresentation[cubes_around_atoms_dim:-cubes_around_atoms_dim,cubes_around_atoms_dim:-cubes_around_atoms_dim,cubes_around_atoms_dim:-cubes_around_atoms_dim]

def get_pyuul_voxels_new(shapes,atoms_channel=None,radius=None,coords=None,pdb_fil = None):
    cubes_around_atoms_dim = 5

    if atoms_channel is None:
        coords, atname = pyuulUtils.parsePDB(pdb_fil)  # get coordinates and atom names
        atoms_channel = torch.zeros((len(atname),len(atname[0])))
        radius = pyuulUtils.atomlistToRadius(atname)  # calculates the radius of each atom
        coords = coords.to(device)
        radius = radius.to(device)
        atoms_channel = atoms_channel.to(device)

    VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel,shapes,resolution = 1,cubes_around_atoms_dim=cubes_around_atoms_dim,function=function)[0,0]

    return VoxelRepresentation[1:,1:,1:]

def get_electrondensity(fil):
    with mrcfile.open(fil) as mrc:
        voxel_size = float(str(mrc.voxel_size).strip("()").split(",")[0])
        voxel = torch.tensor(mrc.data).to(device).permute(2,1,0)

    return voxel,voxel_size

def run_optimization(pdb_file,electrondensity_file,rotate_side_chains=False,out_pdb=False,device="auto",num_optimization_steps=1000,verbose=False,lr=0.0001):
    electrondensity, voxel_size = get_electrondensity(electrondensity_file)
    coords, atname = pyuulUtils.parsePDB(pdb_file)

    if device=="auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    atoms_channel = torch.zeros((len(atname), len(atname[0]))).to(device)
    radius = pyuulUtils.atomlistToRadius(atname).to(device)
    coords = coords.to(device)

    score, new_coords = optimizeCryoEM(coords, atoms_channel, radius, electrondensity.to(device),size_side_voxel=voxel_size, atom_names=atname, out_pdb=out_pdb, num_steps=num_optimization_steps,rotate_side_chains=rotate_side_chains,verbose=verbose,lr=lr)
    return score
