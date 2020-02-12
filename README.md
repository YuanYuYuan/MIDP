# Medical Image Data Pipeline

This project is designed for the data feeding for deep learning models based on PyTorch.
Still under development, use it with caution. 

## Installation

The used python version is 3.7.  And it's recommended to install locally.

```bash
python setup.py install --user
```

## Prepare data

In the following examples, we use the [PDDCA](http://www.imagenglab.com/newsite/pddca/) dataset, and name it as _data_.

## Examples

### Test data loader

```bash
./sample_loader.py --loader-config configs/loader.yaml
```

This program will launch a viewer of the data, you can scroll the mouse wheel to change the slice.

![NAME](./pic/sample_loader.png)

### Test data generator

```bash
./sample_generator.py \
    --loader-config configs/loader.yaml \
    --generator-config configs/generator.yaml
```

The output files are stored in 3D NIfTI (nii.gz) in the _outputs_folder.
One may view these images by [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).

### Generate data list for training

```bash
./generate_data_list.py --loader-config configs/loader.yaml
```

Sample output: _data\_list.yaml_

```yaml
amount:
  test: 0
  total: 3
  train: 2
  valid: 1
list:
  test: []
  train:
  - 0522c0001
  - 0522c0003
  valid:
  - 0522c0002
loader:
- ROIs:
  - BrainStem
  - Parotid_L
  - Parotid_R
  data_dir: data
  data_list: test
  name: PDDCAParser
  preprocess_image: true
```
