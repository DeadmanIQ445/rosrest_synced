import argparse
from pathlib import Path


from detectron2.config.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from utils.validation import FullImageEvalHook
import torch

import os
import glob
from contextlib import suppress
from detectron2.utils.logger import setup_logger
import warnings
warnings.filterwarnings("ignore")
from f1_util import get_f1_score
def main(hparam: argparse.Namespace):
    logger = setup_logger(output='logs/test.txt', name=__name__)
    if hparam.model_path =='':
        return None
    device = (torch.device('cuda') if torch.cuda.is_available()
                               else torch.device('cpu'))
    # TODO: cfg to a separate file or store with model weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.55
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.AUG.ENABLED = True
    cfg.MODEL.WEIGHTS = str(hparam.model_path)
    cfg.MODEL.DEVICE = device.type
    model = build_model(cfg)

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model = model.to(device=device)
    scores = []
    for i in glob.glob('/home/ari/data/ZU/fixed_20220222/test/spltted/*/'):
        input = i[:-1]
        logger.info(input)
        out = os.path.join("/home/ari/data/zu/pred/",input.split('/')[-1])
        pred = FullImageEvalHook(
                model=model,
                eval_period=100,
                current_eval=1,
                sample_size=(1256,1256),
                overlap=0.5,
                # input_dirs=['/home/ari/data/ZU/fixed_20220222/test/spltted/perm_nti_cofp_209'],
                input_dirs=[input],
                output_dir=out)
        pred._do_image_logging()
        try:
            score = get_f1_score(input+'.geojson', out+'.geojson', "vector",True, False,'rtree', logger)
            logger.info(score)
            scores.append(score)
        except AttributeError:
            pass
    logger.info(f'Average F1: {sum(scores)/len(scores)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Eval coco metrics, example: "
        "$ python test.py "
        "   --config config/mask_rcnn_R_50_FPN_3x.yaml "
        "   --dataset /home/shamil/AILab/new_data/dataset/ "
        "   --weights /home/shamil/AILab/Rosreest/weights/aerial_summer_pieceofland.pth")

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='/home/ari/model_1024/model_best.pth')

    args = parser.parse_args()
    main(hparam=args)
