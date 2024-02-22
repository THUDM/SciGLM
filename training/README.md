# Finetune

## 软件依赖

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

#### Finetune

如果需要进行全参数的 Finetune，需要安装 [Deepspeed](https://github.com/microsoft/DeepSpeed)，然后运行以下指令：

```shell
bash finetune_ds3.sh
```

或者：

```
bash /path/to/finetune.sh
```
