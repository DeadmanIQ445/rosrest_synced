import argparse
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.modeling import build_model

from utils.register_dataset import register_dataset


def main(hparam: argparse.Namespace):
    dataset_path, weights_path, config_path = Path(hparam.dataset), Path(hparam.weights), Path(hparam.config)

    assert dataset_path.is_dir()
    assert weights_path.is_file()

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL_WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = hparam.device

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(weights_path))

    register_dataset(dataset_path)

    dataloader = build_detection_test_loader(
        dataset=DatasetCatalog.get('uchastok_test'),
        mapper=DatasetMapper(is_train=False, augmentations=[], image_format='RGB')
    )

    output = inference_on_dataset(
        model,
        dataloader,
        COCOEvaluator(
            dataset_name='uchastok_test',
            tasks=('segm',)
        )
    )

    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Eval coco metrics, example: "
        "$ python test.py "
        "   --config config/mask_rcnn_R_50_FPN_3x.yaml "
        "   --dataset /home/shamil/AILab/new_data/dataset/ "
        "   --weights /home/shamil/AILab/Rosreest/weights/aerial_summer_pieceofland.pth")

    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print(args)

    main(hparam=args)
