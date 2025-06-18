from croz.croz import run_optimization
import os
import numpy as np

def test_on_morph():
    pdb_dataset = "croz/dataset/pdb_models/foldx_models/morph2/"
    test_pdb = "croz/dataset/pdb_models/run1_morph2_job_8.pdb"
    test_map = "croz/dataset/crioEM_data/MORPHOLOGY_2_UNPROCESSED_MAP_cryosparc_P7_J418_Zflipped.mrc"

    print("testing ignorant mode")
    s = run_optimization(test_pdb,test_map,verbose=True)
    print("\tfinal score =", s)

    print("testing optimizing with side chain rotaton mode")
    s = run_optimization(test_pdb,test_map,rotate_side_chains=True,verbose=True)
    print("\tfinal score =", s)

    print("testing saving the pdb in current directiry with the name 'test.pdb' (check it!). ")
    s = run_optimization(test_pdb,test_map,out_pdb="test.pdb",verbose=False)
    print("\tfinal score =", s)

    print("testing optimization with a higher learnign rate (the step of the rotation. bigger steps makes bigger changes per step but might be harder to converge). From now on, no verbosity")
    s = run_optimization(test_pdb,test_map,out_pdb="test_HighLR.pdb",rotate_side_chains=True,verbose=True,lr=0.01)
    print("\tfinal score =", s)

    print("scoring multiple pdb, with limited 'num_optimization_steps' to have it faster (usually less accurate. use verbose=True to see the convergence of the optimization)")
    scores = [ run_optimization(pdb_dataset+t,test_map,verbose=False,num_optimization_steps=100) for t in os.listdir(pdb_dataset)]
    print("\tbest model",np.argmax(scores),"with a score  of ",max(scores))

def run_one_only():
    test_pdb = "croz/dataset/pdb_models/run1_morph2_job_8.pdb"
    test_map = "croz/dataset/crioEM_data/MORPHOLOGY_2_UNPROCESSED_MAP_cryosparc_P7_J418_Zflipped.mrc"
    s = run_optimization(test_pdb,test_map,verbose=True,rotateType="backbone",out_pdb="final.pdb",num_optimization_steps=10000)

run_one_only()