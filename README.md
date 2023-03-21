# TensorRT

```shell
$ CUDA_VISIBLE_DEVICES=${CUDA_NUM} python TensorRT.py
$ tree
├── test.engine
├── test.onnx
├── test_inf.onnx
├── test.pth
└── test.trt
```

![Results](https://user-images.githubusercontent.com/42334717/226349286-c6dbcf24-67ff-459d-8203-6c6b3af27230.png)

||HMean|Precision|Recall|Time|
|:-:|:-:|:-:|:-:|:-:|
|PyTorch|96.756 [%]|96.378 [%]|97.136 [%]|15.478 [ms]|
|TensorRT|96.756 [%]|96.381 [%]|97.135 [%]|1.964 [ms]|
|Difference|0.001 [%p]|0.003 [%p]|-0.001 [%p]|-13.514 [ms]|
|Percentage|0.001 [%]|0.003 [%]|-0.002 [%]|-87.313 [%]|