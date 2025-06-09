# Referencing the Kim's work  for model performance comparison.

Kim's work: MIDI Velocity inference based on Seq2seq and Luong Attention.

Original GitHub Repo of Kim [[link](https://github.com/sappho192/midi-velocity-infer-v2/tree/main)] contains his code and paper, which is presented in the 2023 Autumn meeting of the Acoustical Society of Japan [[Link](https://acoustics.jp/annualmeeting/past-meetings/)]. A great thanks for releasing the code, which makes our experiments more convenient.

## 1. Tested running device

### Ours Env
- Operating System: Linux(Ubuntu 20.04 LTS)
- GPU: NVIDIA RTX 3090 24GB
- NVIDIA Driver: 555.58.02
- CUDA: 12.1
- (Optional) .NET: 8.0.403 (for MIDI-CSV conversion)

## 2. Environment Setup (for demo.ipynb)
- Create Anaconda environment using `environment.yaml` then activate `tf2p39`, OR install the packages like below

```bash
conda create -y --name tf2p39 python==3.9.0
conda activate tf2p39
pip install matplotlib pydot pandas scipy scikit-learn
pip install tensorflow==2.15.0 numpy==1.26.4
pip install notebook
```

## 3. Dataset Preparation
- Option 1: Unzip the "Saarland Music Data (SMD)" and "Maestro-v3" dataset in the converted dataset folder. These MIDI-converted CSV files were prepared by the .NET ver.8.0.403 in my Linux machine. Data statistics are summarized already by the `data_processing.ipynb`.

- Option 2: If want to prepare by yourself - Install .NET with our command, OR (refer this [[Link](https://docs.microsoft.com/ko-kr/dotnet/core/install/linux-ubuntu)]). You can directly build and run the dotnet code we written, OR nano Program.cs to modify our MIDI-CSV conversion code if need. Then, run the `data.ipynb` to summary the data statistics (required by Kim's model inference). Statistics are stored in a .json file.

```bash
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x ./dotnet-install.sh
./dotnet-install.sh --version latest
echo 'export PATH="$HOME/.dotnet:$PATH"' >> ~/.bashrc
source ~/.bashrc

cd "convert_MIDI2CSV"
dotnet new console -o .
dotnet add package NAudio

dotnet build
dotnet run -- "inputFolder" "outputFolder"
# dotnet run -- "/mnt/d/ipynb/dataset/SMD" "/mnt/d/ipynb/Kim/converted_dataset/SMD/test"
```

## 3. Excute the Evaluation
- Run all the code in `seq2seq_evalaution_xxx.ipynb` to load the pretrained model by Kim, and perform the evaluation on the SMD | Maestro-v3 dataset.