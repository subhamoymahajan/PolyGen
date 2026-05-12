from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import pandas as pd
import os
import random
from scipy.spatial.transform import Rotation as R
from rdkit.Geometry import Point3D
import numpy as np

def get_star_neigh(m):
    sneigh=[]
    star=[]
    for atom in m.GetAtoms():
        if atom.GetAtomicNum()==0:
            for neigh in atom.GetNeighbors():
                sneigh.append(neigh.GetIdx())
                star.append(atom.GetIdx())
                break
    return sneigh,star

def build_polymer(monomer, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    sneigh,star = get_star_neigh(monomer)
    Natoms = len(monomer.GetAtoms())
    print(sneigh,Natoms)


    poly = copy.deepcopy(monomer)  # start with a copy

    for i in range(n - 1):
        poly = Chem.CombineMols(poly, copy.deepcopy(monomer))
        poly = Chem.RWMol(poly)
        poly.AddBond( sneigh[1]+i*Natoms, sneigh[0]+(i+1)*Natoms, Chem.BondType.SINGLE)

    for i in range(n):
        for j in range(Natoms):
            atom=poly.GetAtomWithIdx(i*Natoms+j)
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName('M'+str(i+1))
            info.SetResidueNumber(i+1)
            atom.SetMonomerInfo(info)
            atom.SetIntProp("org_id",j)

    for i in range(n-1,-1,-1):
        if i!=n-1:
            poly.RemoveAtom(star[1]+i*Natoms)
        else:
            atom=poly.GetAtomWithIdx(star[1]+i*Natoms)
            atom.SetAtomicNum(1)
        if i!=0:
            poly.RemoveAtom(star[0]+i*Natoms)
        else:
            atom=poly.GetAtomWithIdx(star[0]+i*Natoms)
            atom.SetAtomicNum(1)
    Chem.SanitizeMol(poly)

    return poly

def gen_nmer(smile,outname,n,natoms=None):


    # Example monomer: *-CH2-CH2-*
    monomer = Chem.MolFromSmiles(smile)
    monomer = Chem.AddHs(monomer)

    if natoms is not None:
        n=int( natoms/len(monomer.GetAtoms()) + 0.5)

    #Replace * with H
    monomer_H = copy.deepcopy(monomer)
    sneigh,star = get_star_neigh(monomer_H)
    for idx in star:
        atom=monomer_H.GetAtomWithIdx(idx)
        atom.SetAtomicNum(1)
    success=False
    min_cid = -1
    min_energy = 1E+30
    for i in range(10):
        try:
            params = Chem.AllChem.ETKDGv3()
            seed = random.randint(1, 1_000_000)
            params.randomSeed = seed
            params.pruneRmsThresh = 0.5   # optional: remove near-duplicate conformers

            conf_ids = Chem.AllChem.EmbedMultipleConfs(monomer_H, numConfs=100, params=params)
            for cid in conf_ids:
                Chem.AllChem.UFFOptimizeMolecule(monomer_H, confId=cid)
                ff = AllChem.UFFGetMoleculeForceField(monomer_H, confId=cid)
                energy = ff.CalcEnergy()
                #print('confid',cid,'energy',energy,'kcal/mol')   # kcal/mol
                if energy < min_energy:
                    min_cid = cid
                    min_energy = energy

            success=True
            break
        except:
            pass
    if not success:
        Print('Failed')

    mono_conf = monomer_H.GetConformer(min_cid)
    pos_ref = np.array(mono_conf.GetPositions())
    origin = pos_ref[star[0],:]
    pos_ref -= origin

    #Rotate  to get the other star from v - > (1,0,0)
    end_pos = np.array(pos_ref[star[1],:])
    v = end_pos/ np.linalg.norm(end_pos)
    target = np.array([1.0,0.0,0.0])

    rot,rmsd = R.align_vectors([target],[v])
    pos_ref_rot = rot.apply(pos_ref)

    # Example: 4-mer from *CC*
    poly = build_polymer(monomer, n)

    #Get positions
    all_pos = np.zeros((len(pos_ref_rot)*n,3))
    for ii in range(len(pos_ref_rot)):
        all_pos[ii,: ] = pos_ref_rot[ii,:]

    for i in range(1,n):
        new_start = all_pos[(i-1)*len(pos_ref_rot)+star[1],:]
        old_start = pos_ref_rot[star[0],:]
        trans = np.subtract(new_start,old_start)
        for ii in range(len(pos_ref_rot)):
            all_pos[i*len(pos_ref_rot) + ii, :] = trans+ pos_ref_rot[ii,:]


    conf = Chem.Conformer(poly.GetNumAtoms())

    #Apply to the polymer
    for atom in poly.GetAtoms():
        info = atom.GetPDBResidueInfo()
        resid = info.GetResidueNumber()
        org_id = atom.GetIntProp("org_id")
        idx = atom.GetIdx()
        cur_pos=all_pos[(resid-1)*len(pos_ref_rot)+org_id,:]
        conf.SetAtomPosition(idx, Point3D(float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])))
      
    poly.RemoveAllConformers()
    poly.AddConformer(conf, assignId=True)
    # 4. Optimize geometry
    AllChem.UFFOptimizeMolecule(poly)

    print('outname',outname)
    # 5. Save 3D structure
    if outname[-4:]=='.pdb':
        Chem.MolToPDBFile(poly, outname)
    
    elif outname[-4:]=='.sdf':
        writer = Chem.SDWriter(outname)
        writer.write(poly)
        writer.close()

# E.g.
#df = pd.read_csv('smiles.csv')
#print(df)
#cnt=1
#for index, row in df.iterrows():
#    print(index, row["SMILES"])
#    if not os.path.exists(f'poly{cnt}/poly.sdf'):
#        gen_nmer(row['SMILES'],'poly'+str(cnt)+'/poly.sdf',4,natoms=2000)
#    cnt+=1
