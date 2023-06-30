# NLP-Project

Checkpoint of 3 models: https://husteduvn-my.sharepoint.com/:f:/g/personal/khanh_nd204914_sis_hust_edu_vn/EnOOUMawjH1Gj4Jayag9jQIBDZBt7YFMcJmiNaAxl2g7hA?e=I0qahI

Data after applying backtranslation: https://drive.google.com/drive/folders/1LU4oxv5Eue8jGfPr17IrEkke0EH2BI6c?usp=sharing  

To train the model, and evaluate from a .txt file:
```
git clone https://github.com/lggvu/NLP-Project
pip install -r requirements.txt
python3 src/main.py --mode="train" --decode=<DECODE_SCHEME> --ckpt_name=<PATH_TO_PRETRAINED_CKPT>
python3 evaluate.py -I=<PATH_TO_INPUT_SRC> -L=<REFERENCE FILE> --decode=<DECODE_SCHEME> --ckpt_name=<PATH_TO_PRETRAINED_CKPT>
```
