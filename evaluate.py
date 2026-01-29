import os
import cv2
from tqdm import tqdm
from metrics import EvaluationMetricsV2
import logging


def evaluate(pred_path, dataset):
    pred_root = os.path.join(pred_path, dataset)
    metric = EvaluationMetricsV2()
    mask_root = f'./TestDataset/{dataset}/GT'
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.endswith('.png') or name.endswith('.jpg')]

    for i, mask_name in tqdm(list(enumerate(mask_name_list))):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert pred.shape == mask.shape, f'{pred.shape}: {pred_path}\n{mask.shape}: {mask_path}'
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    return metric_dic


if __name__ == '__main__':
    pred_path = './result/predictions'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p',
                        filename=pred_path + '/eval_log.log',
                        filemode='w')

    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
    logging.info(f'Evaluating...')
    for dataset in datasets:
        metric_dic = evaluate(pred_path, dataset)

        sm = metric_dic['sm']

        emMean = metric_dic['emMean']
        emAdp = metric_dic['emAdp']
        emMax = metric_dic['emMax']

        fmMean = metric_dic['fmMean']
        fmAdp = metric_dic['fmAdp']
        fmMax = metric_dic['fmMax']

        wfm = metric_dic['wfm']
        mae = metric_dic['mae']

        logging.info(f'##### {dataset} #####')
        logging.info(f'sm: {sm}')
        logging.info(f'emMean: {emMean}')
        logging.info(f'emAdp: {emAdp}')
        logging.info(f'emMax: {emMax}')
        logging.info(f'fmMean: {fmMean}')
        logging.info(f'fmAdp: {fmAdp}')
        logging.info(f'fmMax: {fmMax}')
        logging.info(f'wfm: {wfm}')
        logging.info(f'mae: {mae}')
