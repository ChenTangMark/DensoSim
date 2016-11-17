# autoVSimulator

This is a simulator for autonomous driving.

To run the demo,
>> python main.py

Contributors:
Jianyu Chen
Changliu Liu
Zhengyi Zhang

MSC, Berkeley
2016

To start in Windows:
1. Install 32-bit Anaconda2 and set its build-in Python as the default Python.
2. Install Panda3D and set the Anaconda Python as the connected Python: 
(1)Uncheck the python2.7 during installing
(2)Simply create a "panda.pth" file inside your copy of Python, containing 
the path of the panda directory and the bin directory within it on separate 
lines (for example C:\Program Files\Panda3D-1.2.3 and C:\Program Files\Panda3D-1.2.3\bin).
3. Install numpy(using command): pip install numpy
4. Install cvxopt(using command): conda install -c https://conda.anaconda.org/omnia cvxopt
