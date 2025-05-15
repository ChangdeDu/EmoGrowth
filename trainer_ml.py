import sys
import logging
import copy

import pandas as pd
import torch
from utils import factory
from utils.data_manager_ml import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import h5py
def finished(dict,lamda_le,lamda_kd_relation_aff,lamda_kd_relation_data):
    keys = list(dict.keys())[:3]
    values = [dict[key] for key in keys]
    result = np.column_stack(values)
    query = np.array([lamda_le,lamda_kd_relation_aff,lamda_kd_relation_data])
    comparison = np.equal(result,query)
    return np.any(np.all(comparison,axis=1))

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    lambda_le_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2,5,10]
    lamda_kd_relation_data_list = [1]
    lamda_kd_relation_aff_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    if args["init_cls"] == 9:
        a = 'results/' + args['subject'] + '/B9I9'
    elif args["init_cls"] == 3:
        a = 'results/' + args['subject'] + '/B3I3'
    elif args["init_cls"] == 15:
        a = 'results/' + args['subject'] + '/B15I' + str(args["increment"])
    elif args["init_cls"] == 4:
        a = 'results/' + args['subject'] + '/B4I4'
    elif args["init_cls"] == 7:
        a = 'results/' + args['subject'] + '/B7I7'
    elif args['init_cls'] == 16:
        if args['increment'] == 2:
            a = 'results/' + args['subject'] + '/B16I2'
        elif args['increment'] == 3:
            a = 'results/' + args['subject'] + '/B16I3'
    if os.path.exists(a+'/data_sensitivity.csv'):
        df = pd.read_csv(a+'/data_sensitivity.csv')
        dict = df.to_dict(orient='list')
    else:
        dict = {'lamda_le':[],'lamda_kd_ra':[],'lamda_kd_rd':[],'map':[],'macrof1':[],'microf1':[]}
    if args['model_name'] == 'clif':
        for seed in seed_list:
            for lamda_le in lambda_le_list:
                for lamda_kd_relation_data in lamda_kd_relation_data_list:
                    for lamda_kd_relation_aff in lamda_kd_relation_aff_list:
                        if finished(dict,lamda_le,lamda_kd_relation_aff,lamda_kd_relation_data):
                            continue
                        dict['lamda_le'].append(lamda_le)
                        dict['lamda_kd_ra'].append(lamda_kd_relation_aff)
                        dict['lamda_kd_rd'].append(lamda_kd_relation_data)
                        args["seed"] = seed
                        args["device"] = device
                        args["lamda_le"] = lamda_le
                        args["lamda_kd_relation_data"] = lamda_kd_relation_data
                        args["lamda_kd_relation_aff"] = lamda_kd_relation_aff
                        map,macrof1,microf1 = _train(args)
                        dict['map'].append(map)
                        dict['macrof1'].append(macrof1)
                        dict['microf1'].append(microf1)
                        data = pd.DataFrame(data=dict)
                        data.to_csv(a+'/data_sensitivity.csv',index=False)
    elif args['model_name'] == 'agcn':
        for seed in seed_list:
            args["seed"] = seed
            args["device"] = device
            _train(args)
    else:
        # model_name_list = ["finetune","lwf","ewc","replay"]
        model_name_list = ["replay"]
        for seed in seed_list:
            for model_name in model_name_list:
                args["seed"] = seed
                args["device"] = device
                args["model_name"] = model_name
                if model_name == "replay":
                    # args["buffer_type"] = 'random'
                    # _train(args)
                    # args["buffer_type"] = 'rs'
                    # args["device"] = device
                    # _train(args)
                    # args["buffer_type"] = 'ocdm'
                    # args["device"] = device
                    # _train(args)
                    args["buffer_type"] = 'prs'
                    args["device"] = device
                    _train(args)
                else:
                    _train(args)
def data_prepare(args):
    subject = args['subject']
    if args['dataset'] == 'iScience':
        if subject == 'visual':
            sub_data = np.load('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/v_f.npy')
            args['input_size'] = 1000
        else:
            sub_data = np.array(h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/Subject' + subject + '/data_vox8_sub' + subject + '.mat')['data_vox_K'])
            sub_data = sub_data.reshape(-1, 2196)
        sub_data = np.transpose(sub_data).astype('float32')
        if args['init_cls'] == 15:
            if args['increment'] == 3:
                label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/thre_0.1/label_session_b15t4c3.mat')
            elif args['increment'] == 2:
                label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/thre_0.1/label_session_b15t6c2.mat')
        elif args['init_cls'] == 3:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/thre_0.1/label_session_t9c3.mat')
        elif args['init_cls'] == 9:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/iScience/thre_0.1/label_session_t3c9.mat')
        dimension_label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/nips_data/label_dimension.mat')
        dimension_label = np.transpose(dimension_label['label_dimension'])
    elif args['dataset'] == 'PNAS':
        sub_data = np.load('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/feature_res18.npy').astype('float32')
        if args['increment'] == 4:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/label_session_t7c4.mat')
        elif args['increment'] == 7:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/label_session_t4c7.mat')
        elif args['increment'] == 3:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/label_session_b16t4c3.mat')
        elif args['increment'] == 2:
            label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/label_session_b16t6c2.mat')
        dimension_label = h5py.File('/nfs/diskstation/DataStation/KaichengFu/CIL_data/PNAS/affective_rating_USA.mat')
        dimension_label = np.transpose(dimension_label['affective_rating'])
    return sub_data,label,dimension_label

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    data_all = data_prepare(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["init_cls"],
        args["increment"],
        data_all
    )
    model = factory.get_model(args["model_name"], args)

    map_curve = {'map': []}
    hamming_loss_curve = {'hamming_loss':[]}
    avg_precision_curve = {'avg_precision':[]}
    one_error_curve = {'one_error':[]}
    ranking_loss_curve = {'ranking_loss':[]}
    coverage_curve = {'coverage':[]}
    macrof1_curve = {'macrof1':[]}
    microf1_curve = {'microf1':[]}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        if args['model_name'] == 'clif':
            test_map, test_other_metrics = model.eval_multi_label_task(clif=True)
        elif args['model_name'] == "agcn":
            test_map, test_other_metrics = model.eval_multi_label_task(agcn=True)
        else:
            test_map, test_other_metrics = model.eval_multi_label_task()
        test_map = test_map.cpu().detach().numpy()
        model.after_task()
        map_curve['map'].append(test_map)
        hamming_loss_curve['hamming_loss'].append(test_other_metrics[0][1])
        avg_precision_curve['avg_precision'].append(test_other_metrics[1][1])
        one_error_curve['one_error'].append(test_other_metrics[2][1])
        ranking_loss_curve['ranking_loss'].append(test_other_metrics[3][1])
        coverage_curve['coverage'].append(test_other_metrics[4][1])
        macrof1_curve['macrof1'].append(test_other_metrics[5][1])
        microf1_curve['microf1'].append(test_other_metrics[6][1])

        logging.info("Map curve: {}".format(map_curve["map"]))
        logging.info("Hamming_loss curve: {}".format(hamming_loss_curve["hamming_loss"]))
        logging.info("avg_pre curve: {}".format(avg_precision_curve["avg_precision"]))
        logging.info("One_error curve: {}".format(one_error_curve['one_error']))
        logging.info("Ranking_loss curve: {}".format(ranking_loss_curve['ranking_loss']))
        logging.info("Coverage curve: {}".format(coverage_curve['coverage']))
        logging.info("Macrof1 curve: {}".format(macrof1_curve['macrof1']))
        logging.info("Microf1 curve: {}".format(microf1_curve['microf1']))

        print('Average Accuracy: ', sum(map_curve["map"]) / len(map_curve["map"]),
              'Average HL: ', sum(hamming_loss_curve["hamming_loss"]) / len(hamming_loss_curve["hamming_loss"]),
              'Average eAP: ', sum(avg_precision_curve["avg_precision"]) / len(avg_precision_curve["avg_precision"]),
              'Average OneE: ', sum(one_error_curve['one_error']) / len(one_error_curve['one_error']),
              'Average RL: ', sum(ranking_loss_curve['ranking_loss']) / len(ranking_loss_curve['ranking_loss']),
              'Average coverage: ', sum(coverage_curve['coverage']) / len(coverage_curve['coverage']),
              'Average macrof1: ', sum(macrof1_curve['macrof1']) / len(macrof1_curve['macrof1']),
              'Average microf1: ', sum(microf1_curve['microf1']) / len(microf1_curve['microf1']))
        if task == data_manager.nb_tasks-1:
            all_result = np.zeros((8,data_manager.nb_tasks+1))
            all_result[0, 0:data_manager.nb_tasks] = np.array(map_curve['map'])
            all_result[1, 0:data_manager.nb_tasks] = np.array(hamming_loss_curve["hamming_loss"])
            all_result[2, 0:data_manager.nb_tasks] = np.array(avg_precision_curve["avg_precision"])
            all_result[3, 0:data_manager.nb_tasks] = np.array(one_error_curve['one_error'])
            all_result[4, 0:data_manager.nb_tasks] = np.array(ranking_loss_curve['ranking_loss'])
            all_result[5, 0:data_manager.nb_tasks] = np.array(coverage_curve['coverage'])
            all_result[6, 0:data_manager.nb_tasks] = np.array(macrof1_curve['macrof1'])
            all_result[7, 0:data_manager.nb_tasks] = np.array(microf1_curve['microf1'])
            all_result[:,data_manager.nb_tasks] = np.mean(all_result[:,:data_manager.nb_tasks],axis=1)
            if args["init_cls"] == 9:
                a = 'results/' + args['subject']+'/B9I9'
            elif args["init_cls"] == 3:
                a = 'results/' + args['subject']+'/B3I3'
            elif args["init_cls"] == 15:
                a = 'results/' + args['subject'] + '/B15I'+ str(args["increment"])
            elif args["init_cls"] == 4:
                a = 'results/' + args['subject'] + '/B4I4'
            elif args["init_cls"] == 7:
                a = 'results/' + args['subject'] + '/B7I7'
            elif args['init_cls'] == 16:
                if args['increment'] == 2:
                    a = 'results/' + args['subject'] + '/B16I2'
                elif args['increment'] == 3:
                    a = 'results/' + args['subject'] + '/B16I3'
            prefix_file = 'lamda_le_' + str(args["lamda_le"])+'_lamda_kd_relation_aff_' + str(args["lamda_kd_relation_aff"]) + '_lamda_kd_relation_data_' + str(args["lamda_kd_relation_data"])
            if not os.path.isdir(a):
                os.makedirs(a)
            if args["model_name"] == 'clif':
                np.savetxt(a+'/'+args["model_name"]+'_'+prefix_file+'.csv', all_result, delimiter=',')
            elif args["model_name"] == 'replay':
                np.savetxt(a + '/' + args["model_name"] + '_'+ args["buffer_type"] + '.csv', all_result, delimiter=',')
            else:
                np.savetxt(a + '/' + args["model_name"] + '.csv', all_result, delimiter=',')
    if args["model_name"] == 'clif':
        return all_result[0,data_manager.nb_tasks],all_result[6,data_manager.nb_tasks],all_result[7,data_manager.nb_tasks]



def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
