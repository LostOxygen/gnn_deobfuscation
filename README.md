# LaMBdA: Learning-based Deobfuscation of MBA
## Mixed boolean arithmetic deobfuscation using graph neural networks.



## Usage:
```python
python train.py [-h] [--epochs | -e EPOCHS] [-gpu | -g DEVICE_ID] [--lr LR] [--big | -b] [--res | -r]

python test.py [-h] [--gpu | -g DEVICE_ID]
```
## Example Usage:
```python
python train.py --epochs 420 --gpu 2 --lr 0.0001 --big --res

python test.py --gpu -1
```

### Training Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -h, --help | None | shows argument help message |
| -g, --gpu | INT | specifies device ID to use. [0, N] for GPU and -1 for CPU (default=-1)
| -e, --epochs | INT | number of epochs to train (default=1000000) | 
| -bs, --batch_size | INT | batch size (default=1) |
| -lr | FLOAT | learning rate (default=0.00005) |
| -b, --big | BOOL | enables usage of bigger graphs (default=True) |
| -r, --res | BOOL | enables interim result nodes (default=False) |

### Testing Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -h, --help | None | shows argument help message |
| -g, --gpu | INT | specifies device ID to use. [0, N] for GPU and -1 for CPU (default=-1)

