import os, shutil
from datetime import datetime
import fiftyone as fo
from fiftyone import ViewField as F
from detectron2.structures import BoxMode

def get_fiftyone_dicts(samples, labels_dict):
    samples.compute_metadata()
    dataset_dicts = []
    for sample in samples.select_fields(["id", "filepath", "metadata", "segmentations"]):
        if sample.segmentations is not None:
            height = sample.metadata["height"]
            width = sample.metadata["width"]
            record = {}
            record["file_name"] = sample.filepath
            record["image_id"] = sample.id
            record["height"] = height
            record["width"] = width

            objs = []
            #print(sample)
            #print(sample.segmentations)
            #print('NOW DET')
            for det in sample.segmentations.detections:
                #print(det)
                tlx, tly, w, h = det.bounding_box
                bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
                fo_poly = det.to_polyline()
                # very small polygons must be discarded because shapely cant deal with them
                if len(fo_poly.points[0]) < 4:
                    continue
                poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
                poly = [p for x in poly for p in x]
                #print(det.label)
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [poly],
                    "category_id": labels_dict[det.label],
                    #"gt_masks": det.mask
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def clean_instances(instances, confidence_tresh=0.3, size_thresh=0.0):
    # use only detections which are sufficiently confident
    instances_confident = instances[instances.scores > confidence_tresh]
    # constrain the boxes using the image dimensions
    h, w = instances_confident.image_size
    instances_confident.pred_boxes.clip((h, w))
    # discard detections which are too small
    instances_ret = instances_confident[instances_confident.pred_boxes.nonempty(threshold=size_thresh)]
    return instances_ret

def detectron_to_fo(outputs, img_w, img_h, classes, score_tresh=0.3):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    instances = clean_instances(instances, confidence_tresh=score_tresh)
    #print(instances[instances.scores > 0.3])
    #print(instances)
    for pred_box, score, c, mask in zip(
        instances.pred_boxes, instances.scores, instances.pred_classes, instances.pred_masks,
    ):
        x1, y1, x2, y2 = pred_box
        fo_mask = mask.numpy()[int(y1):int(y2), int(x1):int(x2)]
        #print(fo_mask.shape)
        bbox = [float(x1)/img_w, float(y1)/img_h, float(x2-x1)/img_w, float(y2-y1)/img_h]
        #print('Class:', c)
        detection = fo.Detection(label=classes[c.item()], confidence=float(score), bounding_box=bbox, mask=fo_mask)
        if float(score) >= score_tresh:
            #print(bbox)
            detections.append(detection)

    return fo.Detections(detections=detections), instances