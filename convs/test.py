# import pandas as pd
# import csv
# data = pd.read_csv('/data/home/kaicheng/pythonProject/Class_Incremental_Learning/My_code/PyCIL/results/visual/data_sensitivity.csv')
# data = data.iloc[:, 1:]
# # dictionary = data.to_dict(orient='list')
# # data = pd.DataFrame(data=dict)
# data.to_csv('/data/home/kaicheng/pythonProject/Class_Incremental_Learning/My_code/PyCIL/results/visual/data_sensitivity.csv',index=False)
import os
a = os.path.exists('/data/home/kaicheng/pythonProject/Class_Incremental_Learning/My_code/PyCIL/results/3/data_sensitivity.csv')