import py_sod_metrics as metrics
import numpy as np


class EvaluationMetrics():
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255], prediction maps (already sigmoid and times 255)
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure
        emMean = self.EM.get_results()["em"]['curve'].mean()
        # adaptive E-measure
        emAdp = self.EM.get_results()["em"]['adp']
        # F-measure
        # fm = FM.get_results()["fm"]
        # weighted F-measure
        wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,
            'emMean': emMean,
            'emAdp': emAdp,
            'wfm': wfm,
            'mae': mae
        }


class MAE:
    def __init__(self):
        self.mae = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.mae.step(pred=pred, gt=gt)

    def get_results(self):
        return self.mae.get_results()["mae"]
    

class wF:
    def __init__(self):
        self.wfm = metrics.WeightedFmeasure()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.wfm.step(pred=pred, gt=gt)

    def get_results(self):
        return self.wfm.get_results()["wfm"]
    
class meanEM:
    def __init__(self):
        self.em = metrics.Emeasure()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.em.step(pred=pred, gt=gt)

    def get_results(self):
        return self.em.get_results()["em"]['curve'].mean()
    
class meanFM:
    def __init__(self):
        self.fm = metrics.Fmeasure()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.fm.step(pred=pred, gt=gt)

    def get_results(self):
        return self.fm.get_results()["fm"]["curve"].mean()
    
class SM:
    def __init__(self):
        self.sm = metrics.Smeasure()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.sm.step(pred=pred, gt=gt)

    def get_results(self):
        return self.sm.get_results()["sm"]


class adpE:
    def __init__(self):
        self.adpe = metrics.Emeasure()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        self.adpe.step(pred=pred, gt=gt)

    def get_results(self):
        return self.adpe.get_results()["em"]['adp']


class QuickMetrics():
    def __init__(self):
        # self.SM = metrics.Smeasure()
        # self.EM = metrics.Emeasure()
        # self.FM = metrics.Fmeasure()
        # self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        # self.SM.step(pred=pred, gt=gt)
        # self.EM.step(pred=pred, gt=gt)
        # self.FM.step(pred=pred, gt=gt)
        # self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        # sm = self.SM.get_results()["sm"]
        # mean E-measure
        # emMean = self.EM.get_results()["em"]['curve'].mean()
        # adaptive E-measure
        # emAdp = self.EM.get_results()["em"]['adp']
        # F-measure
        # fm = FM.get_results()["fm"]
        # weighted F-measure
        # wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            # 'sm': sm,
            # 'emMean': emMean,
            # 'emAdp': emAdp,
            # 'wfm': wfm,
            'mae': mae
        }


class EvaluationMetricsV2:
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255]
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure and E-measure Curve
        _em = self.EM.get_results()["em"]
        em_curve = np.flip(_em["curve"])
        emMean = _em['curve'].mean()
        emMax = _em['curve'].max()
        emAdp = _em['adp']
        # F-measure curve and PR-curve
        _fm = self.FM.get_results()
        # F-measure
        fm = _fm["fm"]
        fmMean = fm["curve"].mean()
        fmMax = fm["curve"].max()
        fmAdp = fm["adp"]
        fm_curve = np.flip(fm["curve"])
        pr = _fm["pr"]
        p = np.flip(pr["p"])
        r = np.flip(pr["r"])
        # weighted F-measure
        wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,

            'emMean': emMean,
            'emAdp': emAdp,
            'emMax': emMax,
            'em_curve': em_curve,

            'fmMean': fmMean,
            'fmMax': fmMax,
            'fmAdp': fmAdp,
            'fm_curve': fm_curve,

            'wfm': wfm,
            'mae': mae,

            'p': p,
            'r': r
        }
