## 安装

```shell
pip install -r ./requirements.txt
```

> 建议使用虚拟环境，例如`python -m venv ./env`来创建虚拟环境，然后**激活环境**，可参考这里([venv --- 虚拟环境的创建 — Python 3.12.6 文档](https://docs.python.org/zh-cn/3/library/venv.html))。

## 获取测试样本

你可从这里获取[测试样本-阿里云盘](https://www.alipan.com/s/rYymVNLiR1h)。下载完成后，请把压缩文件解压到项目目录下，解压后的项目目录结构如下。

```shell
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d----           9/22/2024  2:46 PM                env
d----           9/22/2024  3:06 PM                models
d----           9/22/2024  2:52 PM                samples
d----           9/22/2024  3:14 PM                src
-a---           9/22/2024  3:37 PM           1132 README.md
-a---           9/22/2024  3:11 PM            788 requirements.txt
```

## 执行演示代码

请在项目根目录下，执行如下命令。

### 车流量分析

[车流量分析演示视频](https://www.bilibili.com/video/BV1sVv8euESQ/)

```shell
python .\src\traffic_analysis.py
```

### 车辆目标跟踪

[车辆目标跟踪演示视频](https://www.bilibili.com/video/BV1Tgv8eGEK3/)

```shell
python .\src\track.py
```

### 俯卧撑监测

[俯卧撑监测演示视频](https://www.bilibili.com/video/BV1sVv8euEAY/)

```shell
python .\src\pushup.py
```

### SSD Mobile Net目标跟踪

[目标跟踪演示视频](https://www.xiaohongshu.com/discovery/item/6686284b000000000a0071b2?source=webshare&xhsshare=pc_web&xsec_token=ABqS51KED1BaRkHIHgU-s70nnm9X7Jtmk6WmOlqAYlo9c=&xsec_source=pc_share)

```shell
python .\src\object_detection_ssd.py
```

### 姿态识别

[姿态识别演示视频](https://www.bilibili.com/video/BV1x3v8erE3h)

```shell
python .\src\pose.py
```

## License

This repository is available under the [GNU V3](https://github.com/CheneyYin/share-cv/blob/master/LICENSE).