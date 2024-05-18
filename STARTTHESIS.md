Инструкция для запуска кода
```
!git clone https://github.com/TradeMaster-NTU/TradeMaster.git
%cd './TradeMaster'
```
```
%%capture
!pip install setuptools==66 
!pip install dtaidistance
!pip install fastdtw
!pip install wheel==0.38.4
!pip install -r requirements.txt
!pip install yapf==0.40.1
```
Для запуска MLP
```
!python /content/TradeMaster/tools/order_execution/train_eteo.py
```
Для запуска Transformer
```
!python /content/TradeMaster/tools/order_execution/train_transformer_eteo.py
```


