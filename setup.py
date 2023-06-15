#-*- encoding: UTF-8 -*-
from setuptools import setup,find_packages

setup(
    name = "datagrabber",          # 包名
    version = "0.2",              # 版本信息
    packages = find_packages('src'),  # 要打包的项目文件夹
    package_dir={'':'src'},       # 这个是拿来重映射包的路径的，如果包跟setup.py不在同一个路径，就需要填写这个参数。
    include_package_data=True,    # 自动打包文件夹内所有数据
    zip_safe=True,                # 设定项目包为安全，不用每次都检测其安全性

    install_requires = [          # 安装依赖的其他包
    ],

    # 设置程序的入口为hello
    # 安装后，命令行执行hello相当于调用hello.py中的main方法
    entry_points={
        'console_scripts':[
            'datagrabber = DataGrabber.launcher:datagrabber'
        ]
     },
 )