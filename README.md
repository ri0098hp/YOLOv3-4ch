yolov3-4ch
==========

# Features
YOLOv3 をRGB-FIR向けに拡張したもの

# TODO
基本的にはissueに投げる

# Installation
0. 必要に応じてCUDA Tool Kitを入れる.
1. cloneする.
  ```bash
  git clone git@github.com:ri0098hp/yolov3-4ch.git
  ```

2. yolov3-4chのフォルダを開きdocker imageをbuildする. 容量は13GBくらいなのでそこそこ時間がかかる.
  ```bash
  ./tools.sh -b
  ```

3. データセットをdatasetフォルダーに入れる. [dataloader](utils/datasets.py) を魔改造してるため次のようなディレクトリ構造推奨...  
  もしくは過去のverから移植すること(当然buildし直す必要あり).
  ```
  <dataset>
  ├── debug
  │   └── 20180903_2040
  │       ├── FIR
  │       ├── FIR_labels
  │       ├── RGB
  │       ├── RGB_crop
  │       └── RGB_raw
  │   
  ├── fujinolab-all
  │   ├── 20180903_1113
  │   │   ├── FIR
  │   │   ├── FIR_labels
  │   │   ├── RGB
  │   │   ├── RGB_crop
  │   │   └── RGB_raw
  │   └── 20190116_2008
  │       ├── FIR
  │       ├── FIR_labels
  │       ├── RGB
  │       ├── RGB_crop
  │       ├── RGB_labels
  │       └── RGB_raw
  │   
  └── kaist-all
      ├── train
      │   ├── FIR
      │   ├── FIR_labels
      │   └── RGB
      └── val
          ├── FIR
          ├── FIR_labels
          ├── RGB
          └── RGB_labels
  ```

4. 必要に応じてオンラインで学習状況を確認できる [wandb](https://wandb.ai/home) に登録してログインキーを登録する. 詳細は[公式レポ](https://github.com/ultralytics/yolov5/issues/1289)参照.  
今まで通りtensor boradを使うなら[次の起動時](#起動)に次のコマンドを実行.
```bash
wandb off
```
# Usage
## 通常モード
  ### 起動
  次のコマンドを実行.
  ```bash
  ./tools.sh -b
  ```

  ### データセットの準備
  [ここ](data/fujinolab-all.yaml)を参考にディレクトリとクラスを指定.  
  なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある (RGBを基準に出ディレクトリを置換している)

  ### 訓練
  [ここ](memo.txt)を参照. 基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
  以下例...
  ```bash
  python train.py --data [.yamlへのパス] --batch-size [n^2 (自動推定:-1)] --epochs [エポック数]
  ```

  ### テスト
  準備中

  ### 検知
  準備中

## デバッグモード
準備中
