## 打包python

1. 确保有__init__.py
2. 确保有MANIFEST.in, setup.py
3. 按照格式配置好文件
4. 运行 ```python setup.py sdist bdist_wheel```
5. 安装方法 ```pip install dist/test_tools-0.1-py3-none-any.whl --upgrade```
