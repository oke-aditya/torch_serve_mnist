# Torch Serve

## Installing Torch Serve: -
- Refer to Torch serve [docs](https://github.com/pytorch/serve#install-torchserve)
- Install both pytorch, torchvision and of course torch serve.

## Tutorial how to deploy with Torch-Serve
- Tutorial on how to run app using Torch-Serve
- Most code is taken torchserve library from PyTorch.
- Trains a simple MNIST Digit Classifer and deploy using TorchServe.

Torch Serve provide a simple API to create a backend API for ML Models.

Steps to take Model to Torch Serve: -
- Create a `src/model.py` file which defines the model
- Train the model. I have provided the training script in `src/train.py`.
- Write a custom Handler to preprocess the data to recongize the image.

Now Use Torch Serve.

1. `torch-model-archiver --model-name mnist --version 1.0 --model-file src/model.py --serialized-file src/model.pt --handler src/handler.py`

2. Register the model on TorchServe using the above model archive file and run digit recognition inference.

```
mkdir model_store
mv mnist.mar model_store/
torchserve --start --model-store model_store --models mnist=mnist.mar
curl http://127.0.0.1:8080/predictions/mnist -T ./examples/0.png
```
