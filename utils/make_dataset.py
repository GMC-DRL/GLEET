from problems.cec_dataset import *

'''for augmented cec2021 dataset generation'''

pro_name=['Bent_cigar','Schwefel','bi_Rastrigin','Grie_rosen','Hybrid','Hybrid','Hybrid','Composition','Composition','Composition','Mix21']
def get_config(problem_id):
    pro=pro_name[problem_id-1]
    subproblems=None
    sublength=None
    Comp_sigma=None
    Comp_lamda=None
    indicated_dataset=None
    if problem_id==5:
        # hybrid 1
        subproblems=['Schwefel','Rastrigin','Ellipsoidal']
        sublength=[0.3,0.3,0.4]
    elif problem_id==6:
        # hybrid 2
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel']
        sublength=[0.2,0.2,0.3,0.3]
    elif problem_id==7:
        # hybrid 3
        subproblems=['Escaffer6','Hgbat','Rosenbrock','Schwefel','Ellipsoidal']
        sublength=[0.1,0.2,0.2,0.2,0.3]
    elif problem_id==8:
        # # composition 1
        subproblems=['Rastrigin','Griewank','Schwefel']
        sublength=None
        Comp_sigma=[10,20,30]
        Comp_lamda=[1,10,1]
    elif problem_id==9:
        # composition 2
        subproblems=['Ackley','Ellipsoidal','Griewank','Rastrigin']
        sublength=None
        Comp_sigma=[10,20,30,40]
        Comp_lamda=[10,1e-6,10,1]
    elif problem_id==10:
        # composition 3
        subproblems=['Rastrigin','Happycat','Ackley','Discus','Rosenbrock']
        sublength=None
        Comp_sigma=[10,20,30,40,50]
        Comp_lamda=[10,1,10,1e-6,1]
    elif problem_id==11:              
        indicated_dataset=cec2021
    return (pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset)

def make_dataset(dim, num_samples, batch_size,max_x,min_x,problem_id, shifted=True, rotated=True, biased=False):
    '''return a dataloader'''
    pro,subproblems,sublength,Comp_sigma,Comp_lamda,indicated_dataset=get_config(problem_id)
    return Training_Dataset(filename=None, dim=dim, num_samples=num_samples, problems=pro, biased=biased, shifted=shifted, rotated=rotated,
                            batch_size=batch_size,xmin=min_x,xmax=max_x,indicated_specific=True,indicated_dataset=indicated_dataset,
                            problem_list=subproblems,problem_length=sublength,Comp_sigma=Comp_sigma,Comp_lamda=Comp_lamda)
