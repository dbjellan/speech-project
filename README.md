
### Instructions
1. Install Python 2.7 and Pip
2. Install Pip Dependancies
```
pip install numpy scrapy tensorflow jupyter matplotlib scipy
pip install https://pypi.python.org/packages/b0/d8/d9babf3e4fa3ac8094e1783415bf60015a696779f4da4c70ae6141aa5e3a/scikits.audiolab-0.11.0.tar.gz#md5=f93f17211c7763d8631e0d10f37471b0
```
3. Download/Decrompress Dataset from Voxforge (Warning will take up ~20Gb)
```
cd tools
scrapy runspider spider.py
```
4. Run Jupyter Notebook
```
cd src 
jupyter notebook SpeechProject.ipynb
```