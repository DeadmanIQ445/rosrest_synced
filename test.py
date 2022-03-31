import argparse
from pathlib import Path


from detectron2.config.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from utils.validation import FullImageEvalHook
import torch

import warnings
warnings.filterwarnings("ignore")

def main(hparam: argparse.Namespace):
    if hparam.model_path =='':
        return None
    device = (torch.device('cuda') if torch.cuda.is_available()
                               else torch.device('cpu'))
    # TODO: cfg to a separate file or store with model weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = str(hparam.model_path)
    cfg.MODEL.DEVICE = device.type
    model = build_model(cfg)

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model = model.to(device=device)
    pred = FullImageEvalHook(
            model=model,
            eval_period=100,
            current_eval=1,
            sample_size=(1024,1024),
            input_dirs=['test_inp'],
            output_dir="/home/ari/test_image")
    pred._do_image_logging()

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

    print(args)

    main(hparam=args)
