from setuptools import setup

setup(name='prediction',
      version='1.0',
      description='Python package for CRISPR/Cas9 single guide RNA on- and off-target activities prediction',
      url='https://github.com/Peppags/CRISPRont-CRISPRofft',
      author='Guishan Zhang',
      author_email='gszhang@stu.edu.cn',
      packages=['prediction'],
      py_modules=['prediction.crispr_offt_prediction', 'prediction.crispr_offt_prediction_batch', 'prediction.crispr_on_prediction',
                   'prediction.crispr_ont_prediction_batch'],
      platform='Ubuntu 20.04.1 LTS',
      install_requires=['numpy==1.19.5', 'pandas==1.1.5', 'tensorflow==2.4.0', 'keras==2.3.0'],
      )
