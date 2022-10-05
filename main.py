'''
基于网格的改进DBSCAN方法
所需数据文件格式为txt，样例文件(鸢尾花数据集)为iris_data.txt
'''
import numpy as np
from collections import deque

# 扫描半径
EPS=0.95
# 最小包含点数
MIN_PTS=8
# 调整网格边长的参数，LPG=EPS/P,P为整数
P=2

# 数据文件路径
DATA_PATH=r"./iris_data.txt"

def load_data(file_path):
    '''载入数据'''
    return np.loadtxt(file_path,dtype=float,delimiter=" ")

def get_dist(a:np.ndarray,b:np.ndarray)->float:
    """计算两样本点欧式距离"""
    return np.sqrt(((a-b)**2).sum())

def is_neibour(data2grid:list,a:int,b:int,ex_grid:list):
    '''判断点b是否在点a的eps矩形邻域'''
    if(a==b):
        return False
    core_grid=data2grid[a]
    test_grid=data2grid[b]
    # print(core_grid,test_grid,P)
    for i in range(len(core_grid)):
        if(test_grid[i]<(core_grid[i]-ex_grid)
            or test_grid[i]>(core_grid[i]+ex_grid)):
            return False

    return True

def init():
    '''初始化聚类所需的各类信息'''
    data=load_data(DATA_PATH)
    size=len(data) #样本数
    lpg=EPS/P #网格边长
    # lpr=(2*P+1)*lpg if P>1 else 3*lpg  #网格矩形eps边长
    ex_grid=P if P>1 else 1 #计算核心点时向四周扩展网格的数目
    visited=[0]*len(data) #记录各样本点是否被访问，0未访问、1访问、-1噪声
    #记录样本中各个特征的边界
    border=zip(np.min(data,axis=0),np.max(data,axis=0)) 
    #向右端补齐边界为整数个lpg
    border=[(i[0],i[0]+((i[1]-i[0])//lpg+1)*lpg) for i in border]
    # 样本点到网格的映射
    data2grid=[[int((d[i]-border[i][0])/lpg) for i in range(len(d))] for d in data]
    
    # print(data2grid)
    return data,data2grid,visited,ex_grid,size


def scan()->list:
    '''进行聚类'''
    
    data,data2grid,visited,ex_grid,size=init()
    result=[] #聚类结果
    seeds=deque([]) #种子点队列

    for i in range(size):
        if(visited[i]):
            continue
        seeds.append(i)
        temp=[] #存储一个簇的所有点
        while(len(seeds)>0):
            
            crt=seeds.popleft()
            visited[crt]=1
            temp.append(crt) #将该点加入簇
            
            count=0
            neibours=[]
            for i in range(size):
                if(is_neibour(data2grid,crt,i,ex_grid)):
                    count+=1
                    neibours.append(i)
            # eps矩形邻域的点不超过MIN_PTS
            if(count<MIN_PTS):
                visited[crt]=-1
                continue
            count=0
            points=[]
            for n in neibours:
                if get_dist(data[n],data[crt])<=EPS:
                    count+=1
                    points.append(n)
            # eps半径的点不超过MIN_PTS
            if(count<MIN_PTS):
                visited[crt]=-1
                continue
            # 将eps半径邻域点加入seeds
            seeds.extend([i for i in points if not visited[i] and i not in seeds])

        result.append(temp.copy())
    
    # 以样本点索引形式返回,同时返回噪点
    return result,visited

if __name__=='__main__':
    res=scan()
    print(res[0])
    
    # a=np.array([[1,2],[3,4]])
    # print(np.max(a,axis=0))
    # with open("./iris.txt") as f:
    #     iris=f.readlines()
    #     data=[" ".join(line.split(" ")[1:5])+"\n" for line in iris]
    #     labels=[line.split(" ")[5][1:-2]+"\n" for line in iris]
        
    #     with open("./iris_data.txt","w") as f1:
    #         f1.writelines(data)
    #     with open("./iris_label.txt","w") as f2:
    #         f2.writelines(labels)
        
    