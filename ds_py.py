import array
import numpy as np

def get_numpy():
    try:
        # 6.3
        arr = array.array('b', [1, 2, 3, 4, 5])
        arr[1] = 6
        length = len(arr)
        arr = np.array([range(i, i+3) for i in [1,3,545]], dtype=np.string_)
        arr = np.linspace(0, 1, 5)
        arr = np.zeros(10,dtype='float')
        arr = np.ones((3,5),dtype='int64')
        arr = np.full((2,4),55.5,dtype='float32')
        arr = np.arange(2,10,2)
        arr = np.random.random((3,4))
        arr = np.random.normal(1,2,(2,6))
        arr = np.random.randint(1,9,(2,3))
        arr = np.eye(4,6)
        arr = np.empty((3,7))
        x = np.array([1, 2, 3])
        m = x[np.newaxis,:]

        np.random.seed(44)
        arr_1 = np.random.randint(10,size=6)
        arr_2 = np.random.randint(10,size=(4,3))
        arr_3 = np.random.randint(10,size=(3,2,2))

        print(arr_3.ndim)
        print(arr_3.shape)
        print(arr_3.size)
        print(arr_3.dtype)
        print(arr_3.itemsize)
        print(arr_3.nbytes)
        '''
        多维数组的索引方式与列表的列表索引方式是不同的。
        列表的列表在Python中需要使用多个中括号进行索引，如`x[i][j]`的方式。
        '''
        arr_2[0,:]
        arr_2[0,-2]
        '''
        请记住，与Python的列表不同，NumPy数组是固定类型的。
        这意味着，如果你试图将一个浮点数值放入一个整数型数组，
        这个值会被默默地截成整数。这是比较容易犯的错误。
        '''
        '''
        如果是获取行数据的话，可以省略后续的切片，写成更加简洁的方式：
        '''
        print(arr_1[0])  # 等同于 x2[0, :]
        '''
        一个非常重要和有用的概念你需要知道的就是数组的切片返回的实际上是
        子数组的*视图*而不是它们的副本。这是NumPy数组的切片和Python列表的切片的
        主要区别，列表的切片返回的是副本。
        '''
        '''
        进行连接的数组如果具有不同的维度，使用`np.vstack`（垂直堆叠）和
        `np.hstack`（水平堆叠）会更加清晰
        '''
        arr_1[1] = 99.33
        arr_1[-2::-2]
        arr_2[:2,:3]
        arr_2[::-2,:2]
        arr_2[::-1,:]
        arr_2_copy = arr_2.copy()
        arr_2[3,1] = 99
        arr_4 = np.arange(0,12).reshape(4,3)
        arr_1.reshape(3,2)
        arr_11 = np.random.randint(6,size=2)
        np.concatenate([arr_1,arr_11])
        np.concatenate([arr_2,arr_4],axis=1)
        arr_1 = np.random.random(3)
        arr_0 = np.random.random((4,1))
        np.hstack([arr_2,arr_0])
        x1,x2,x3 = np.split(arr_1,[1,3])
        first,second,third = np.hsplit(arr_2,[1,2])
        p1,p2 = np.vsplit(arr_2,[2])
        # 6.4
        import numpy as np
        seed = np.random.seed(44)
        values = np.random.randint(1, 10, size=5)
        %timeit (1 / values)
        m = np.arange(5)
        values / m
        x = np.arange(9).reshape(3,3)
        x ** 2
        x + 2 
        x - 2 
        x * 2
        x / 2
        x // 2
        -x
        x**2
        x % 2
        abs(-((-x * 5 - 10) ** 2))
        theta = np.linspace(0, 2*np.pi, 3)
        np.sin(theta)
        np.cos(theta)
        np.tan(theta)
        np.arccos(theta)
        np.arcsin(theta)
        np.arctan(theta)
        np.exp(values)
        np.exp2(values)
        np.power(3,values)
        np.log(values)
        np.log2(values)
        np.log10(values)
        x = [0, 0.001, 0.01, 0.1]
        np.expm1(x)
        np.log1p(x)
        '''
        还有当输入值很小时，可以保持精度的指数和对数函数：
        x = [0, 0.001, 0.01, 0.1]
        print("exp(x) - 1 =", np.expm1(x))
        print("log(1 + x) =", np.log1p(x))
        '''
        from scipy import special
        special.gamma(values)
        special.gammaln(values)
        special.beta(values, 2)
        special.erf(values)
        special.erfc(values)
        special.erfinv(values)
        x = np.arange(5)
        y = np.zeros(10)
        np.multiply(x,10,out=y[::-2])
        '''
        对于二元运算ufuncs来说，还有一些很有趣的聚合函数可以
        直接从数组中计算出结果。例如，如果你想`reduce`一个数组，
        你可以对于任何ufuncs应用`reduce`方法。reduce会重复在数组的
        每一个元素进行ufunc的操作，直到最后得到一个标量。
        '''
        np.add.reduce(values)
        np.multiply.reduce(values)
        '''
        如果你需要得到每一步计算得到的中间结果，
        你可以调用`accumulate`：
        np.add.accumulate(x)
        '''
        np.add.accumulate(values)
        np.multiply.accumulate(values)
        y = np.arange(1,6)
        np.multiply.outer(values,values)
        x = np.arange(1, 6)
        np.multiply.outer(y, y)
        # 6.5
        import numpy as np
        seed = np.random.seed(44)
        l = np.random.random(1000000)
        %timeit sum(l)
        %timeit np.sum(l)
        m = np.random.rand(3,3)
        m.sum()
        '''
        这里的`axis`参数指定的是*让数组沿着这个方向进行压缩*，
        而不是指定返回值的方向。因此指定`axis=0`意味着第一个维度将被压缩：
        对于一个二维数组来说，就是数组将沿着列的方向进行聚合运算操作
        '''
        np.sum(m,axis=0)
        np.max(m,axis=1)
        '''
        美国总统的平均身高？
        '''
        import pandas as pd
        df = pd.read_csv(r'F:\git\notebooks\data\president_heights.csv')
        df.head(5)
        heights = np.array(df['height(cm)'])
        print(heights)

        heights.mean()
        heights.std()
        heights.max()
        heights.min()
        np.percentile(heights,25)
        np.median(heights)
        np.percentile(heights,75)

        %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set_theme()

        plt.hist(heights)
        plt.title('Height Distribution of US Presidents')
        plt.xlabel('height(cm)')
        plt.ylabel('number')
        plt.show()
        # 6.6
        import numpy as np
        a = np.array([1,2,3])
        b = np.array([5,5,5])
        a + b
        a + 5

        M = np.ones((3,3))
        M + a

        a = np.arange(3)
        b = np.arange(3)[:,np.newaxis]
        a + b 
        # 6.7
        '''
        在NumPy中应用广播不是随意的，而是需要遵从严格的一套规则：
        - 规则1：如果两个数组有着不同的维度，维度较小的那个数组
        会沿着最前（或最左）的维度进行扩增，扩增的维度尺寸为1，
        这时两个数组具有相同的维度。
        - 规则2：如果两个数组形状在任何某个维度上存在不相同，
        那么两个数组中形状为1的维度都会广播到另一个数组对应唯独的
        尺寸，最终双方都具有相同的形状。
        - 规则3：如果两个数组在同一个维度上具有不为1的不同长度，
        那么将产生一个错误。
        '''
        M + a
        a = np.arange(3).reshape((3, 1))
        b = np.arange(3)
        M = np.ones((3,2))
        a = np.arange(3)
        M + a
        a = a[:,np.newaxis]
        a + M
        '''
        实际上广播可以应用到*任何*的二元ufunc上。
        例如下面我们采用`logaddexp(a, b)`函数求值，
        这个函数计算的是$\log(e^a + e^b)$的值，
        使用这个函数能比采用原始的exp和log函数进行计算得到
        更高的精度
        '''
        np.logaddexp(M,a)
        x = np.random.random((10,3))
        mean = x.mean(axis=0)
        x_center = x - mean
        x_center.mean(0)
        '''
        二维分布 (bivariate distribution)是同时考虑两个随机变量的
        情况，表示特性值或特性值组与相应频率 (或频数) 之间的对应
        关系，或者是同时考虑的两个随机变量取给定值或属于一个给定
        值集的 概率分布 所确定的函数称为二维分布 。
        '''
        x = np.linspace(0,5,50)
        y = np.linspace(0,5,50)[:,np.newaxis]
        z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

        %matplotlib inline
        import matplotlib.pyplot as plt

        plt.imshow(z, cmap='viridis',origin='lower',extent=[0,5,0,5])
        plt.colorbar()
        # 6.12
        import numpy as np
        import pandas as pd

        '''
        pandas 的 DataFrame对象，知道它们由三个组成并存储为属性的组件很有用：
            .values：对应的二维NumPy值数组。
            .columns：列索引：列名称。
            .index：行的索引：行号或行名。
        '''
        rainfall = pd.read_csv(r'C:\Users\陈泽鹏\Desktop\拆分合并\Python_Data_Science_Handbook\notebooks\data\Seattle2014.csv')['PRCP'].values
        inches = rainfall / 254
        inches.shape

        %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set_theme()

        plt.hist(inches,bins=40)
        x = np.array([1,2,3,4,5])
        x < 3
        x == 3
        (x ** 2) == (x * 2)
        rng = np.random.RandomState(44)
        x = rng.randint(10,size=(3,4))
        '''
        在Python中，`False`实际上代表0，而`True`实际上代表1
        '''
        np.count_nonzero(x < 6)
        np.sum(x<6,axis=1)
        np.any(x<6)
        np.all(x<6)
        np.all(x<6,axis=0)
        np.sum((inches > 0.5) & (inches < 1))
        print("无雨的天数： ", np.sum(inches == 0))
        np.sum(inches != 0)
        np.sum(inches > 0.5)
        np.sum((inches > 0) & (inches < 0.2))
        x[x<6]

        rainy = inches > 0
        days = np.arange(365)
        summer = ((days > 172) & (days < 262))

        np.median(inches[rainy])
        np.median(inches[summer])
        np.max(inches[summer])
        np.median(inches[rainy & ~summer])
        '''
        `and`和`or`对整个对象进行单个布尔操作，
        而`&`和`|`会对一个对象进行多个布尔操作（比如其中每个二进制位）。
        对于NumPy布尔数组来说，需要的总是后两者。
        '''
        X = np.arange(12).reshape((3, 4))
        row = np.array([0, 1, 2])
        col = np.array([2, 1, 3])
        mask = np.array([1, 0, 1, 0], dtype=bool)
        import numpy as np
        np.random.seed(42)
        x = np.random.randn(100) # 获得一个一维100个标准正态分布值

        # 得到一个自定义的数据分组，区间-5至5平均取20个点，每个区间为一个数据分组
        bins = np.linspace(-5, 5, 20)
        counts = np.zeros_like(bins) # counts是x数值落入区间的计数

        # 使用searchsorted，得到x每个元素在bins中落入的区间序号
        i = np.searchsorted(bins, x)

        # 使用at和add，对x元素在每个区间的元素个数进行计算
        np.add.at(counts, i, 1)
        '''
        Maria plays college basketball and wants to go pro. Each season she maintains a record of her play. 
        She tabulates the number of times she breaks her season record for most points and least points in a game.
        Points scored in the first game establish her record for the season, and she begins counting from there.
        enumerate() 是一个内置函数，它用来迭代容器（如列表、元组、字符串等）时同时获取每个元素的索引位置和该元素的值。
        这个函数非常有用，因为它简化了在迭代过程中同时获取元素和其索引的流程。
        使用方法
        enumerate() 函数的基本语法如下：
        python复制代码
        enumerate(iterable, start=0)
        iterable：是一个序列、迭代器，或其他支持迭代的对象。
        start：是一个可选参数，用来指定索引的起始值，默认为 0。
        '''
#         def breakingRecords(scores):
#         # Write your code here
#             low, high = [], []
#             for i, score in enumerate(scores):
#                 if i > 0:
#                     if score > max(scores[:i]):
#                         high.append(score)
#                     elif score < min(scores[:i]):
#                         low.append(score)
#             return len(high),len(low)   
        # 6.13
        import numpy as np
        rand = np.random.RandomState(10)

        x = rand.randint(100,size=10)
        ind = [0,4,8]
        x[ind]
        '''
        当使用高级索引时，结果数组的形状取决于*索引数组*的形状
        而不是*被索引数组*的形状
        '''
        ind = np.array([[2,5],[6,9]])
        x[ind]
        X = np.arange(12).reshape(3,4)
        row = np.array([0,1,2])
        col = np.array([2,1,3])
        X[row,col]
        '''
        这里，每个行索引都会匹配每个列的向量，
        就像我们在广播的算术运算中看到一样
        '''
        X[row[:,np.newaxis],col]
        '''
        记住高级索引结果的形状是*索引数组广播后的形状*而不是被索引数组形状，
        这点非常重要
        '''
        X[2,[2,0,1]]
        X[1:,[2,0,1]]
        mask = np.array([1,0,1,0],dtype=bool)
        X[row[:,np.newaxis],mask]

        mean = [0,0]
        cov = [[1,2],[2,5]]
        X = rand.multivariate_normal(mean,cov,100)
        X.shape

        %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set_theme

        plt.scatter(X[:,0],X[:,1])
        '''
        np.random.choice(5, 6, replace=True)#可以看到有相同元素
	    array([3, 4, 1, 1, 0, 3])
        np.random.choice(5, 6, replace=False)#会报错，因为五个数字中取六个，
        不可能不取到重复的数字
	    ValueError: Cannot take a larger sample than population when 'replace=False'
        '''
        '''
        首先需要知道，对于二维张量，shape[0]代表行数，shape[1]代表列数，
        同理三维张量还有shape[2]
        一般来说，-1代表最后一个，所以shape[-1]代表最后一个维度，如在二维张量里，
        shape[-1]表示列数，注意，即使是一维行向量，shape[-1]表示行向量的元素总数，
        换言之也是列数
        '''
        indice = np.random.choice(X.shape[0],20,replace=False)
        selection = X[indice]
        selection.shape
        plt.scatter(X[:,0],X[:,1],alpha=0.3)
        plt.scatter(selection[:,0],selection[:,1],s=200,facecolor='None',alpha=0.8)

        x = np.arange(10)
        i = np.array([2,1,8,4])
        x[i] = 99

        x = np.zeros(10)
        x[[0,0]] = [[4,6]]
        i = [2, 3, 3, 4, 4, 4]
        x[i] += 1
        '''
        因为`x[i] += 1`是操作`x[i] = x[i] + 1`的简写，
        而`x[i] + 1`表达式的值已经计算好了，然后才被赋值给`x[i]`。
        因此，上面的操作不会被扩展为重复的运算，而是一次的赋值操作
        '''
        '''
        `at()`方法不会预先计算表达式的值，而是每次运算时实时得到，
        方法在一个数组`x`中取得特定索引`i`，
        然后将其取得的值与最后一个参数`1`进行相应计算，
        这里是加法`add`。还有一个类似的方法是`reduceat()`方法
        '''   
        x = np.zeros(10)
        np.add.at(x,i,1)

        np.random.seed(42)
        x = np.random.randn(100)

        bins = np.linspace(-5,5,20)
        counts = np.zeros(20)

        i = np.searchsorted(bins,x)
        np.add.at(counts,i,1)
        '''
        ds='steps':这是一个可选参数，表示绘制图形时使用的线条样式。
        在这里，它设置为'steps',表示使用阶梯状的线条。
        其他可选值包括'lines'(默认值，表示使用实线连接数据点)和'dashed'(表示使用虚线连接数据点)。
        '''
        plt.plot(bins,counts,ds='steps')
        plt.hist(x,bins,histtype='step')
        np.histogram(x,bins)
        '''
        上面的结果说明当涉及到算法的性能时，永远不可能是一个简单的问题。
        对于大数据集来说一个很高效的算法，并不一定也适用于小数据集，
        反之亦然（参见[大O复杂度](02.08-Sorting.ipynb#Aside:-Big-O-Notation)）。
        我们这里使用自己的代码实现这个算法，目的是理解上面的基本函数，
        后续读者可以使用这些函数构建自己定义的各种功能。
        在数据科学应用中使用Python编写代码的关键在于，
        你能掌握NumPy提供的很方便的函数如`np.histogram`，你也能知道什么情况下适合使用它们，
        当需要更加定制的功能时你还能使用底层的函数自己实现相应的算法。
        '''
        import numpy as np
        rand = np.random.RandomState(44)
        X = rand.rand(10, 2)
        dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        differences.shape
        sq_differences = differences ** 2
        sq_differences.shape
        dist_sq = sq_differences.sum(-1)
        dist_sq.shape
        k = 2
        nearest_partition = np.argpartition(dist_sq, k + 1, axis=1)
        # 创建2个列表
        m = [1, 2, 3]
        n = [4, 5, 6]
        z = list(zip(m,n))
        print("*zip(m, n)返回:", *zip(m, n))
        m2, n2 = zip(*zip(m, n))
        print("m2和n2的值分别为:", m2, n2)
        # 若相等，返回True；说明*zip为zip的逆过程
        print(m == list(m2) and n == list(n2))

        x_1 = [0.74464048, 0.36050084]
        x_2 = [0.35931084, 0.60923838]
        m, n = zip(x_1,x_2)
        m, n = zip(*zip(x_1,x_2))

        import numpy as np
        def bogosort(x):
            while np.any(x[:-1] > x[1:]):
                np.random.shuffle(x)
            return x
        
        l = np.array([2,5,1,-1,100,-20])
        bogosort(l)
        '''
        对数组进行排序，返回排序后的结果，不改变原始数组的数据，你应该使用`np.sort`
        '''
        np.sort(l)
        '''
        如果你期望直接改变数组的数据进行排序，你可以对数组对象使用它的`sort`方法：
        '''
        l.sort()
        '''
        相关的函数是`argsort`，它将返回排好序后元素原始的序号序列：
        '''
        m = np.array([2,5,1,-1,100,-20])
        i = np.argsort(m)
        '''
        结果的第一个元素是数组中最小元素的序号，
        第二个元素是数组中第二小元素的序号，以此类推。
        这些序号可以通过高级索引的方式使用，从而获得一个排好序的数组：
        译者注：更好的问题应该是，假如我们希望获得数组中第二、三小的元素，
        我们可以这样做
        '''
        m[i[1:3]]
        ra = np.random.RandomState(44)
        l = ra.randint(0,10,(4,6))
        np.sort(l,axis=0)
        np.sort(l,axis=1)
        '''
        必须注意的是，这样的排序会独立的对每一行或者每一列进行排序。
        因此结果中原来行或列之间的联系都会丢失
        '''   
        '''
        有时候我们并不是需要对整个数组排序，
        而仅仅需要找到数组中的*K*个最小值。
        NumPy提供了`np.partition`函数来完成这个任务；结果会分为两部分，
        最小的*K*个值位于结果数组的左边，而其余的值位于数组的右边，顺序随机
        '''
        m = np.array([2,5,1,-1,100,-20,2002])
        np.partition(m,3)
        '''
        每个分区内部，元素的顺序是任意的
        '''
        ra = np.random.RandomState(44)
        l = ra.randint(0,10,(4,6))
        np.partition(l,3,axis=1)

        X = np.random.rand(10,2)

        %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn;seaborn.set_theme()
        plt.scatter(X[:,0],X[:,1],s=100)

        dis_sq = np.sum((X[np.newaxis,:,:] - X[:,np.newaxis,:])**2,axis=-1)
        '''
        检查这个矩阵的对角线元素，对角线元素的值是点与其自身的距离平方，应该全部为0
        '''
        np.diagonal(dis_sq)
        np.argsort(dis_sq,axis=1)
        '''
        最左边的列就会给出每个点的最近邻，结果中的第一列是0到9的数字：
        这是因为距离每个点最近的是自己
        '''
        k = 2
        nearest_partition = np.argpartition(dis_sq,k+1,axis=1)

        plt.scatter(X[:,0],X[:,1],s=100)
        '''
        count = 0
        for i in range(X.shape[0]):
            for j in nearest_partition[i,:(k+1)]:
                print(count,X[i],X[j])
                count+=1

        x_1 = np.array([2,3,4])
        x_2 = np.array([6,7,8])
        m = zip(x_1,x_2)
        for i in m:
            print(i)
        '''
        '''
        x = [1, 2, 3, 4, 5]
        y = [10, 15, 13, 18, 16]

        # 绘制线图，并自定义外观
        plt.plot(
            x,                         # X轴数据
            y,                         # Y轴数据
            marker='o',                # 标记样式：圆点
            linestyle='-',             # 线条样式：实线
            color='green',              # 线条颜色：蓝色
            linewidth=2,               # 线宽：2
            markersize=10,              # 标记大小：8
            label='数据1'               # 图例标签
        )
        ————————————————
        '''
        for i in range(X.shape[0]):
            for j in nearest_partition[i,:(k+1)]:
                plt.plot(*zip(X[i],X[j]),color='black')

        def migratoryBirds(arr):
            '''
            d = {}
            _list = ['a','b','c']
            _list_1 = [1,2,3]
            for i in range(len(_list)):
                d[_list[i]] = _list_1[i]
            '''
            l = []
            '''
            set 是无序的，不能像列表那样通过索引访问。
            要解决这个问题，可以将 set 转换为 list,
            然后再进行操作
            '''
            arr_ = list(set(arr))
            for j in arr_:
                count = 0 
                for i in arr:
                    if j == i:
                        count+=1
                l.append(count)
                '''
                使用max()函数找到列表中的最大值，
                然后我们用list.index()方法找到这个最大值在列表中的索引
                '''
            return arr_[l.index(max(l))]

        import numpy as np
        name = ['apple','m.c','banana','pear']
        age = [23,45,12,80]
        weight = [23.1,25.6,77.2,78.9]
        '''
        结构化数组可以用来存储复合的数据类型
        '''
        x = np.zeros(4,dtype=int)
        data = np.zeros(4,dtype={'names':('name','age','weight'),
                                'formats':('U10','i4','f8')})
        data.dtype
        '''
        `U10`代表着“Unicode编码的字符串，最大长度10”，
        `i4`代表着“4字节（32比特）整数”，
        `f8`代表着“8字节（64比特）浮点数”
        '''
        data['name'] = name
        data['age'] = age
        data['weight'] = weight
        print(data)

        data['name']
        data[0]
        data[-1]['name']
        data[data['age']<30]['name']
        '''
        如果你想要完成的工作比上面的需求还要复杂的话，
        你应该考虑使用Pandas包，下一章的主要内容。我们将会看到，
        Pandas提供了`Dataframe`对象，它是一个在NumPy数组的基础上构建的结构，提供了很多有用的数据操作功能，包括上面结构化数组的功能
        '''

        np.dtype({'names':('name','age','weight'),
                'formats':('U10','i4','f8')})
        
        np.dtype({'names':('name','age','weight'),
                'formats':((np.str_,10),int,'float64')})
        
        np.dtype([('name','S10'),('age','i4'),('weight','f8')])
        '''
        如果类型的名称并不重要，你可以省略它们，
        你甚至可以在一个以逗号分隔的字符串中指定所有类型：
        '''
        np.dtype('S10,i4,f8')

        tp = np.dtype([('id','i8'),('mat','f8',(3,3))])
        X = np.zeros(1,dtype=tp)
        X[0]
        X['mat'][0]
        '''
        为什么不用一个多维数组或者甚至是Python的字典呢？
        原因是NumPy的`dtype`数据类型直接对应这一个C语言的结构体定义，
        因此存储这个数组的内容内容可以直接被C语言的程序访问到。
        如果你在写访问底层C语言或Fortran语言的Python接口的话，
        你会发现这种结构化数组很有用
        '''
        data['age']
        data_rec = data.view(np.recarray)
        data_rec.age

        %timeit data['age']
        %timeit data_rec['age']
        %timeit data_rec.age
        '''
        是使用更方便简洁的写法还是使用更高性能的写法，取决于你应用的需求
        '''
    except Exception as e:
        print(e)
    return

def get_pandas():
    try:
        data = 123
        print(f'data is {data}')
    except Exception as e:
        print(e)
