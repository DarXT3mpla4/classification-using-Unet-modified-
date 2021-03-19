# modifiedUNET

### To run the program

- There are parameters used to train the program. You can look in the arguments of train_skip.py file. Use different values of batchsize (1,2,4), learning-rate (examples 0.001, 0.0001, 0.0005) and use different epoch values (10, 100, 300) and then record the result.
```
python train_skip.py
```

### To run tensorboard to visualize results
- Once training process starts, *runs* and *checkpoints* folders are created. Run following command to run tensorboard. Save the training and validation loss graphs along with the crack images.
```
tensorboard --logdir=[location of folder inside runs] --port=[port number]
```