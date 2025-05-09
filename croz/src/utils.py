import torch

from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

atom_description_hash = {
						"batch": 0,
						"chain": 1,
						"resnum": 2,
						"resname": 3,
						"at_name": 4,
						"alternative": 5,
						"alternative_conf": 6
						}

def min_max_scale(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    scaled = (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    return scaled

def find_rigid_alignment(A, B):
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

def get_rotation_matrix(theta_x, theta_y, theta_z):
    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)

    # Rotation matrices
    a1 = torch.tensor([1, 0, 0], device=cos_x.device, dtype=cos_x.dtype).unsqueeze(0)
    a2 = torch.stack([torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype), cos_x, -sin_x]).unsqueeze(0)
    a3 =torch.stack([torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype), sin_x, cos_x]).unsqueeze(0)

    Rx = torch.cat([
        a1,
        a2,
        a3
    ])

    Ry = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ], device=theta_x.device, dtype=theta_x.dtype)

    a1 = torch.stack([cos_y, torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype), sin_y]).unsqueeze(0)
    a2 = torch.tensor([0, 1, 0], device=cos_x.device, dtype=cos_x.dtype).unsqueeze(0)
    a3 = torch.stack([-sin_y, torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype),cos_y]).unsqueeze(0)

    Ry = torch.cat([
        a1,
        a2,
        a3
    ])



    a1 = torch.stack([cos_z,-sin_z,torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype)]).unsqueeze(0)
    a2 = torch.stack([sin_z,cos_z,torch.tensor(0, device=cos_x.device, dtype=cos_x.dtype)]).unsqueeze(0)
    a3 = torch.tensor([0,0, 1], device=cos_x.device, dtype=cos_x.dtype).unsqueeze(0)

    Rz = torch.cat([
        a1,
        a2,
        a3
    ])
    # Combined rotation: First X, then Y, then Z (R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx

    return R

def writePDB(coords,atom_names,pdbOut = "outpdb.pdb"):

    atom_names = atom_names[0]
    coords = coords[0].tolist()  # shape [N, 3], converted to list of lists
    structure = Structure('example')
    model = Model(0)

    for i, (full_name, coord) in enumerate(zip(atom_names, coords), start=1):
        resname, resnum, atomname, chainid = full_name.split('_')
        resnum = int(resnum)
        atomname = atomname.strip()

        if not model.has_id(chainid):
            model.add(Chain(chainid))
        chain = model[chainid]

        resid = (' ', resnum, ' ')
        if not chain.has_id(resid):
            residue = Residue(resid, resname, '')
            chain.add(residue)
        else:
            residue = chain[resid]

        element = atomname[0] if atomname[0].isalpha() else 'X'  # crude guess
        atom = Atom(atomname, coord, 1.0, 1.0, ' ', atomname, i, element=element)
        residue.add(atom)

    structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbOut)