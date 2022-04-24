#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:28:35 2021

@author: johnviljoen

This script will find the approximate max min values of all the lookup tables 
by iterating through them manually. I am aware this is very much not the best
way to go about this, but in the interest of time...

going with 100 points in every dimension for every lookup table now

I can feel a computer science student rolling in his grave as I write this 
monstrosity.
"""

from tables.c_tables import table_wrap
import torch
from tqdm import tqdm

points = 100
alpha = torch.linspace(-20,90,points)
beta = torch.linspace(-30,30,points)
el = torch.linspace(-25,25,points)

gt = torch.zeros(points)
out = torch.zeros(points)

list1 = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
list2 = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
list3 = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
list4 = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
list5 = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
list6 = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
list7 = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']

sum_list = list1 + list2 + list3 + list4 + list5 + list6 + list7
sum_list = ['delta_Cm_ds']

lim_dict = {}

for coeff in sum_list:
    table = table_wrap(coeff)
    
    storage = torch.zeros([points, points, points])
    
    # sweep across coeff table in every dimension
    for i, alpha_val in tqdm(enumerate(alpha)):
        for j, beta_val in enumerate(beta):
            for k, el_val in enumerate(el):
                inp = torch.cat([alpha_val.unsqueeze(0),beta_val.unsqueeze(0),el_val.unsqueeze(0)])
                
                # print(k)
                storage[i,j,k] = table.call(inp)
                
    lim_dict[coeff] = torch.tensor([torch.max(storage), torch.min(storage)])
   
torch.save(lim_dict, 'tables/NNs/lookup_table_limits.pt')

