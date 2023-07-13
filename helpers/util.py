import os, shutil
from datetime import datetime
import fiftyone as fo
from fiftyone import ViewField as F

# in - dataset: fiftyone dataset (or view)
#      labels: str naming a class or list of str naming classes, or None for identity
#      mode: str how to process labels ('include' or 'exclude')
# out - dataset_ret: fiftyone dataset view
#                    -> (include mode)  only including labels or
#                    -> (exclude mode) including all classes BUT labels 
def select_classes(dataset, labels=None, mode='include'):
    if labels is None:
        return dataset
    if type(labels) is str:
        labels = [labels]
    if mode == 'include':
        dataset_ret = dataset.filter_labels("detections", F("label").is_in(labels))
        dataset_ret = dataset_ret.filter_labels("segmentations", F("label").is_in(labels))
    elif mode == 'exclude':
        dataset_ret = dataset.filter_labels("detections", ~F("label").is_in(labels))
        dataset_ret = dataset_ret.filter_labels("segmentations", ~F("label").is_in(labels))
    else:
        raise NotImplementedError
    return dataset_ret

# in - dataset: fiftyone dataset (or view)
#      superclasses: str naming a superclass or list of str naming superclasses, or None for identity
#      labels: str or list of [str, list of str] specifying labels belonging to each superclass
# out - dataset_ret: fiftyone dataset view with merged/renamed labels
def merge_to_superclass(dataset, superclasses, labels):
    if superclasses is None:
        return dataset
    if type(superclasses) is str:
        superclasses = [superclasses]
    if type(labels) is str:
        labels = [labels]
    else:
        labels = [[label] if type(label) is str else label for label in labels]
    assert len(superclasses) == len(labels)
    dataset_ret = dataset
    for superclass, class_labels in zip(superclasses, labels):
        mapping = {label: superclass for label in class_labels}
        dataset_ret = dataset_ret.map_labels("detections", mapping)
        dataset_ret = dataset_ret.map_labels("segmentations", mapping)
    return dataset_ret

# in - datasets: a list of fiftyone datasets
# out - dataset_ret: a dataset containing all input datasets
def concat_datasets(datasets):
    dataset_ret = fo.Dataset()
    for dataset in datasets:
        for sample in dataset:
            dataset_ret.add_sample(sample)
    return dataset_ret

def get_timestamp():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    tmp = datetime.fromtimestamp(ts)   
    str_tmp = str(tmp)
    str_tmp = str_tmp[:-10]
    return str_tmp[:10] + '_' + str_tmp[11:]

def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)