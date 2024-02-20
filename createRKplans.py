from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir
import sys

pool_op_kernel_sizes_2D = [
[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
[[4, 4], [2, 2], [1, 1]],
[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
[[2, 2], [2, 2]],
[[4, 4], [1, 1], [2, 2], [2, 2]],
[[2, 2], [2, 2], [2, 2]],
[[2, 2], [2, 2], [2, 2], [2, 2]],
[[2, 2], [2, 2], [2, 2], [2, 2], [1, 1]],
[[2, 2], [2, 2], [2, 2], [4, 4]],
[[2, 2], [2, 2], [2, 2], [2, 2]]
]

TASK_NAME = ['Task001_BrainTumour', 'Task002_Heart', 'Task003_Liver', 'Task004_Hippocampus', 'Task005_Prostate', 'Task006_Lung', 'Task007_Pancreas', 'Task008_HepaticVessel', 'Task009_Spleen', 'Task010_Colon']

if len(sys.argv) > 3:
    task_name = str(sys.argv[1]) # e.g. 'Task003_Liver'
    plans_fname = join(preprocessing_output_dir, task_name, '%s%s%s' % ('nnUNetPlansv2.1_plans_', sys.argv[2], 'D.pkl'))
    plans = load_pickle(plans_fname)

    TYPE = ['E', 'I', 'R']
    plans['RK_type'] = TYPE.index(sys.argv[3])
    pool_str = ""

    stage = list(plans['plans_per_stage'].keys())[0]
    stage_plans = plans['plans_per_stage'][stage]
    id = TASK_NAME.index(task_name)

    if 2 == int(sys.argv[2]):
        stage_plans['pool_op_kernel_sizes'] = pool_op_kernel_sizes_2D[id]
        num_pool = len(pool_op_kernel_sizes_2D[id])

        net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        stage_plans['conv_kernel_sizes'] = net_conv_kernel_sizes[:(num_pool+1)]
        pool_str = '_pool' + str(num_pool)

    plans['base_num_features'] = int(sys.argv[4])
    file_name = join(preprocessing_output_dir, task_name, '%s%s%s%s%s%s%s%s' % ('nnUNetPlansv2.1_RK-', sys.argv[3], '_', sys.argv[4], pool_str, '_plans_', sys.argv[2], 'D.pkl'))

    save_pickle(plans, file_name)
    print(file_name)
else:
    print('python createRKplans.py TASK_NAME DIMENSION RK_TYPE NUM_FEATURES')
