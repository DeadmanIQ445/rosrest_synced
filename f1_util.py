from typing import List, Callable
from multiprocessing import Pool

import geojson
import numpy as np

from PIL import Image
from shapely.geometry import Polygon, asShape, MultiPolygon, Point
from shapely.wkb import dumps, loads

import rtree
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, f1_score
from rasterio.warp import transform_geom
import rasterio
from rasterio.features import shapes


from shapely.geometry import shape
import json

def get_geojson_from_tif(tif):
    with rasterio.Env():
        with rasterio.open(tif) as src:
            image = src.read(1, out_dtype="uint8") # first band

            mask = src.dataset_mask()

            results = ({'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform)))
            results = {"features": list(results)}

            results["crs"] = src.crs.to_string()
            return results


def convert_tif_to_geojson(input_tif, output_geojson):
    with open(output_geojson, "w+") as dst:
        results = get_geojson_from_tif(input_tif)
        dst.write(json.dumps(results))
        
        
IOU_THRESHOLD = 0.5

global_groundtruth_polygons = []
global_groundtruth_rtree_index = rtree.index.Index()

def _has_match_basic(polygon_serialized):

    polygon = loads(polygon_serialized)
    best_iou = 0
    for label in global_groundtruth_polygons:
        metric = iou(polygon, label)
        if metric > best_iou:
            best_iou = metric
        if best_iou > IOU_THRESHOLD:
            break

    if best_iou > IOU_THRESHOLD:
        return True
    else:
        return False


def _has_match_rtree(polygon_serialized):

    polygon = loads(polygon_serialized)
    best_iou = 0
    candidates = [
        loads(candidate_serialized)
        for candidate_serialized
        in global_groundtruth_rtree_index.intersection(
            polygon.bounds, objects='raw'
        )
    ]
    for candidate in candidates:
        metric = iou(polygon, candidate)
        if metric > best_iou:
            best_iou = metric

    if best_iou > IOU_THRESHOLD:
        return best_iou
    else:
        return best_iou


def iou(polygon1: Polygon, polygon2: Polygon):
    # buffer(0) may be used to “tidy” a polygon
    # works good for self-intersections like zero-width details
    # http://toblerity.org/shapely/shapely.geometry.html
    poly1_fixed = polygon1.buffer(0)
    poly2_fixed = polygon2.buffer(0)
    # return poly1_fixed.buffer(0).intersection(poly2_fixed).area / poly2_fixed.area
    return poly1_fixed.buffer(0).intersection(poly2_fixed).area / poly1_fixed.union(poly2_fixed).area


def get_polygons(json) -> List[Polygon]:
    res = []  # type: List[Polygon]
    if isinstance(json['crs'], str):
        src_crs = json['crs']
    else:
        src_crs = json['crs']['properties']['name']
    dst_crs = 'EPSG:4326'
    for f in json["features"]:
        if isinstance(f["geometry"], geojson.MultiPolygon):
            new_geom = transform_geom(src_crs=src_crs,
                                      dst_crs=dst_crs,
                                      geom=f["geometry"])
            if new_geom['type'] == 'Polygon':
                res += [asShape(new_geom)]
            else:
                res += [asShape(geojson.Polygon(c)) for c in new_geom["coordinates"]]
        elif isinstance(f["geometry"], geojson.Polygon):
            new_geom = transform_geom(src_crs=src_crs,
                                      dst_crs=dst_crs,
                                      geom=f["geometry"])
            res += [asShape(new_geom)]
        else:
            raise Exception("Unexpected FeatureType:\n" + f["geometry"]['type'] + "\nExpected Polygon or MultiPolygon")
    return res


def pixelwise_f1_score(groundtruth_array, predicted_array, v: bool=False, echo=print):
    assert groundtruth_array.shape == predicted_array.shape, "Images has different sizes"

    groundtruth_array[groundtruth_array > 0] = 1
    groundtruth_binary = groundtruth_array.flatten()
    predicted_array[predicted_array > 0] = 1
    predicted_binary = predicted_array.flatten()

    return f1_score(groundtruth_binary, predicted_binary)


def objectwise_f1_score(groundtruth_polygons: List[Polygon],
                        predicted_polygons: List[Polygon],
                        method='rtree',
                        v: bool=False,
                        multiproc: bool=True,
                        logger=None):
    """
    Measures objectwise f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb

    :param groundtruth_polygons: list of shapely Polygons;
    we suppose that these polygons are not intersected
    :param predicted_polygons: list of shapely Polygons with
    the same size as groundtruth_polygons;
    we suppose that these polygons are not intersected
    :param method: 'rtree' or 'basic'
    :param v: is_verbose
    :param multiproc: nables/disables multiprocessing
    :return: float, f1-score
    """

    if method == 'basic':
        global global_groundtruth_polygons
        global_groundtruth_polygons.extend(groundtruth_polygons)
        # for some reason builtin pickling doesn't work

        if multiproc:
            tp = sum(Pool().map(_has_match_basic, (dumps(polygon) for polygon in predicted_polygons)))
        else:
            tp = sum(map(_has_match_basic, (dumps(polygon) for polygon in predicted_polygons)))
    elif method == 'rtree':
        global global_groundtruth_rtree_index
        # for some reason builtin pickling doesn't work
        for i, polygon in enumerate(groundtruth_polygons):
            global_groundtruth_rtree_index.insert(
                i, polygon.bounds, dumps(polygon)
            )
        if multiproc:
            tp = sum(Pool().map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))
        else:
            tp = sum(map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))
    else:
        raise Exception('Unknown method: ' + method)
    # to avoid zero-division
    a = list(map(_has_match_rtree, (dumps(polygon) for polygon in predicted_polygons)))
    if logger:
        logger.info(f"Accuracy: {sum(a)/len(a)}")
    else:
        print(sum(a)/len(a))
    if tp == 0:
        return 0.
    fp = len(predicted_polygons) - tp
    fn = len(groundtruth_polygons) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if logger:
        logger.info(f"Precision: {precision}, Recall: {recall}")
    else:
        print(precision, recall)

    return 2 * (precision * recall) / (precision + recall)


def get_f1_score(groundtruth_path, predicted_path, format_, v, multiproc, method,logger=None):

    if format_ == 'raster':
        groundtruth_image = Image.open(groundtruth_path)

        predicted_image = Image.open(predicted_path)

        groundtruth_array = np.array(groundtruth_image)
        predicted_array = np.array(predicted_image)

        #we reshape the predicted mask to the gt mask size
        if predicted_array.shape != groundtruth_array.shape:
            predicted_array = resize(predicted_array, groundtruth_array.shape, 0).astype(np.uint8)

        score = pixelwise_f1_score(groundtruth_array, predicted_array, v=v)
        
        
    elif format_ == 'vector':
        gt = geojson.load(open(groundtruth_path))

        pred = geojson.load(open(predicted_path))

        gt_polygons = get_polygons(gt)
        pred_polygons = get_polygons(pred)
        score = objectwise_f1_score(gt_polygons, pred_polygons, v=v, method=method, multiproc=multiproc, logger=logger)
        
    return score

if __name__ == "__main__":
    # print(get_f1_score("/home/ari/data/ZU/fixed_20220222/test/Razmetka2/perm_nti_cofp_311.geojson","/home/ari/data/ZU/predictions/perm_nti_cofp_311.geojson", "vector",True, False,'rtree'))
    print(get_f1_score("/home/ari/data/ZU/fixed_20220222/test/Razmetka2/Утяково_ЦОФП.geojson","/home/ari/data/ZU/predictions/Утяково_ЦОФП.geojson", "vector",True, False,'rtree'))
    # print(get_f1_score("/media/deadman445/disk/ff/zu_347000_6194000_ЦОФП.geojson","/media/deadman445/disk/ff/555.geojson", "vector",True, True,'rtree'))