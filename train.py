import argparse
from datetime import datetime
from pathlib import Path
import os

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from utils.io_helpers import get_num_train_image
from utils.register_dataset import register_dataset
from utils.trainer import CustomTrainer
import logging
import warnings
warnings.filterwarnings("ignore")


def get_num_iterations(epochs: int, batch_size: int, num_train: int) -> int:
    return epochs * num_train // batch_size

def loggers():
    # logging.getLogger('detectron2.utils.events').setLevel(logging.ERROR)
    logging.getLogger('detectron2.checkpoint.c2_model_loading').setLevel(logging.ERROR)
    logging.getLogger('detectron2.data.common').setLevel(logging.ERROR)
    logging.getLogger('detectron2.engine.defaults').setLevel(logging.ERROR)
    logging.getLogger('detectron2.data.build').setLevel(logging.ERROR)
    logging.getLogger('fvcore.common.checkpoint').setLevel(logging.ERROR)


def main(hparam: argparse.Namespace):
    loggers()
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    assert hparam.config.is_file(), f"Config does not exists: {str(hparam.config)}!"

    cfg = get_cfg()
    cfg.PATCH = (hparam.size, hparam.size)
    cfg.merge_from_file(cfg_filename=str(hparam.config))
    cfg.DATALOADER.NUM_WORKERS = hparam.num_workers
    if hparam.weights != '':
        assert Path(hparam.weights).is_file()
        logger.info("Weights enabled!")
        cfg.MODEL.WEIGHTS = str(hparam.weights)
    cfg.SOLVER.BASE_LR = hparam.lr
    cfg.SOLVER.IMS_PER_BATCH = hparam.batch_size
    cfg.SOLVER.MAX_ITER = get_num_iterations(epochs=hparam.epochs,
                                             batch_size=hparam.batch_size,
                                             num_train=get_num_train_image(hparam.dataset_path))
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER//(hparam.epochs//hparam.epoch_log)

    logger.info(f"Max iter: {cfg.SOLVER.MAX_ITER}, Eval Period: {cfg.TEST.EVAL_PERIOD}")
    register_dataset(hparam.dataset_path)

    cfg.AUGMENT = True if hparam.augment else False

    # All experiments wii be stores in ./output folder
    current_time = str(datetime.now()).replace(".", "_").replace(":", "_").replace(" ", "_")
    config_name = str(hparam.config).split(r"/")[1]
    folder_name = f"{cfg.PATCH[0]}_{current_time}_{config_name}_{hparam.batch_size}_{hparam.lr}_{str(hparam.augment)}_{hparam.epochs}_cosine_lr"
    output_folder = Path("./output") / folder_name
    os.makedirs(output_folder, exist_ok=True)

    cfg.OUTPUT_DIR = str(output_folder)
    logger.info(f"Saving folder: {cfg.OUTPUT_DIR}")

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # if hparam.weights:
    #     trainer.checkpointer.load(str(hparam.weights))
    # trainer.scheduler.milestones = cfg.SOLVER.STEPS

    

    trainer.train()

    # cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.WEIGHTS = str(output_folder / "model_best.pth")
    cfg.DATASETS.TEST = ('uchastok_test',)

    coco_final_evaluation = output_folder / "coco-eval-final"
    if not coco_final_evaluation.is_dir():
        coco_final_evaluation.mkdir()

    evaluator = COCOEvaluator('uchastok_test', ("segm",), False, output_dir=str(coco_final_evaluation))
    val_loader = build_detection_test_loader(cfg, 'uchastok_test')
    output = inference_on_dataset(trainer.model, val_loader, evaluator)
    logger.info(output)

    with open(output_folder / "final_results.txt", 'w') as file:
        file.write(str(output))

    logger.info("Training is finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "PANet instance segmentation, example of usage: "
        "$ CUDA_VISIBLE_DEVICES=0 python train.py --config config/mask_rcnn_R_50_FPN_3x.yaml --augment")

    parser.add_argument("--config", type=Path, help="Path to config file", default="config/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument('--augment', action='store_true', help='Enable augmentation', default=False)
    parser.add_argument('--dataset_path', type=Path, default='/home/ari/fixed_20220222/dataset', help="Path to dataset")
    parser.add_argument('--weights', type=str, default='', help="Resume training with pretrained weights.")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=4)
    parser.add_argument('--epochs', type=int, help="Num epochs", default=300)
    parser.add_argument('--num_workers', type=int, help="Num workers", default=4)
    parser.add_argument('--size', type=int, help="Tile size", default=1024)
    parser.add_argument('--epoch_log', type=int, help="Log every N epochs", default=10)

    args = parser.parse_args()
    main(hparam=args)
