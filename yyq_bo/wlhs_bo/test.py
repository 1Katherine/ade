# 由于ipconfig/all在windows中是查看ip地址
# 所以将此命令运行在os.system中，即可查看系统的ip地址等信息
import os
# os.system('ipconfig/all')
# 因为python file_name.py可以直接执行py文件
# 所以可以通过os.system来执行py代码
import os
# 获取当前文件所属的文件夹路径
# pwd = os.path.dirname(os.path.abspath(__file__))
# code = pwd + '\\wLHS_Bayesian_Optimization.py'
# print(code)
# os.system('python ' + code)

for i in range(20):
    os.system('python wLHS_Bayesian_Optimization.py')