# DS3000-25fall
everything about the course DS3000 in 2025fall

# Plant Disease Detection

## 1) Setup 
```bash
# Python 3.10
py -3.10 -m pip install -U pip
py -3.10 -m pip install torch torchvision
# optional (TensorFlow path if you run tf code)
py -3.10 -m pip install "tensorflow>=2.16,<2.18"

# download & unzip into ./data
kaggle datasets download -d karagwaanntreasure/plant-disease-detection -p data -w
Expand-Archive -Path "data/plant-disease-detection.zip" -DestinationPath "data" -Force
# if an extra 'Dataset/' folder appears:
#   Move-Item data/Dataset/* data/ -Force; Remove-Item data/Dataset -Recurse -Force
Remove-Item "data/plant-disease-detection.zip"


Tasks assign:

1. Jackie
2. Jim
3. Edward
4. Jackie & Jim
5. Desmond
6. Aspha & Edward
