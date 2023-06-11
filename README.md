##  VCNet

### Prepare data
Preparation: Put the images with the same id in one folder. You may use
```
python prepare.py
```

### Train Model
```
python train_2020.py --name vcnet  --warm_epoch 5 --droprate 0 --stride 1 --erasing_p 0.5 --autoaug --inputsize 384 --lr 0.02  --gpu_ids 0,1 --batchsize 24;
```

### Test Model
```
python test_2020.py --name vcnet --gpu_ids 0,1
```

### Evaluation
```
python evaluate_multi.py
```