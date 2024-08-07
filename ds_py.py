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

        import numpy as np
        import pandas as pd

        '''
        Pandas的`Series`是一个一维的带索引序号的数组，可以通过列表或数组进行创建
        '''
        data = pd.Series([2,2.5,4.57,8.77])
        data
        '''
        `Series`封装了一个值的序列（由列表指定）和一个索引序号的序列，
        我们可以分别通过`values`和`index`属性访问它们。
        `values`属性就是你已经很熟悉的NumPy数组
        '''
        data.values
        data.index
        data[1]
        data[1:3]
        '''
        显式定义的索引提供了`Series`对象额外的能力。
        例如，索引值不需要一定是个整数，
        可以用任何需要的数据类型来定义索引显式定义的索引提供了`Series`对象额外的能力。
        例如，索引值不需要一定是个整数，可以用任何需要的数据类型来定义索引
        '''
        data = pd.Series([2,2.5,4.57,8.77],index=['a','b','c','d'])
        data['b']
        '''
        我们亦可以使用非连续的或非序列的索引值
        '''
        data = pd.Series([2,2.5,4.57,8.77],index=[2,'b',6,0])
        data[6]

        population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
        population = pd.Series(population_dict)
        data['California']
        '''
        下面这个操作是字典所不具有的，`Series`还支持按照数组方式的操作来对字典进行切片
        '''
        data['California':'Illinois']

        '''
        `data`可以是一个列表或NumPy数组，在这种情况下`index`默认是一个整数序列
        '''
        pd.Series([12,23,34])
        '''
        `data`可以是一个标量，这种情况下标量的值会填充到整个序列的index中
        '''
        pd.Series(5,index=[100,200,'moon'])
        '''
        `data`可以是一个字典，这种情况下`index`默认是一个排序的关键字key序列
        每种情况下，index都可以作为额外的明确指定索引的方式，
        结果也会依据index参数而发生变化
        '''
        pd.Series({1:'a', 2:'b', 34:'a'},index=[2,34])
        '''
        上例表明，结果中包含的数据仅是index明确指定部分
        '''
        '''
        `DataFrame`既可以被当成是一种更通用的NumPy数组，
        也可以被当成是一种特殊的Python字典
        '''
        area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
        area = pd.Series(area_dict)
        
        states = pd.DataFrame({'population': population, 'area': area})
        states
        '''
        `DataFrame`对象也像`Series`一样有着`index`属性，包括所有的数据的索引标签
        '''
        states.index
        '''
        它额外含有一个`columns`属性，同样也是一个`Index`对象，存储这所有列的标签
        '''
        states.columns
        '''
        `DataFrame`将一个列标签映射成一个`Series`对象，里面含有整列的数据。
        例如，访问`area`属性会返回一个`Series`对象包含前面我们放入的面积数据
        '''
        states['area']
        '''
        这里要注意一下容易混淆的地方：NumPy的二维数组中，
        `data[0]`会返回第一行数据，而在`DataFrame`中，
        `data['col0']`会返回第一列数据。正因为此，
        最好还是将`DataFrame`当成是一个特殊的字典而不是通用的二维数组
        '''
        states[states.columns[0]]

        pd.DataFrame(population,columns=['population'])
        pd.DataFrame(population)

        data = pd.DataFrame({'a': i, 'b': i**2} for i in range(3))
        data = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
        # data = pd.DataFrame({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) 实际运行时后一个字典被识别为index，即行索引

        '''
        在给定一个二维NumPy数组的情况下，我们指定其相应的列和行的索引序列来构建一个`DataFrame`。
        如果行或列的index没有指定，默认会使用一个整数索引序列来指定
        '''
        data = pd.DataFrame(np.random.rand(3,2),index=['a','foo','op'],columns=[1,'bb'])

        A = np.zeros(3, dtype=[('A','i8'), ('B','f8')])
        A = pd.DataFrame(A)

        '''
        `Series`和`DataFrame`对象都包含着一个显式定义的*索引index*对象，
        它的作用就是让你快速访问和修改数据。`Index`对象是一个很有趣的数据结构，
        它可以被当成*不可变的数组*或者*排序的集合*
        （严格来说是多数据集合，因为`Index`允许包含重复的值）
        '''
        ind = pd.Index([1,23,4,5])
        ind[1]
        ind[::2]
        print(ind.size,ind.shape,ind.ndim,ind.dtype)
        '''
        NumPy数组和`Index`对象的最大区别是你无法改变`Index`的元素值，它们是不可变的
        这种不变性能在多个`DataFrame`之间共享索引时提供一种安全性，
        避免因为疏忽造成的索引修改和其他的副作用
        '''
        ind[1] = 2
        '''
        Pandas对象被设计成能够满足跨数据集进行操作，
        例如连接多个数据集查找或操作数据，这很大程度依赖于集合运算。
        `Index`对象遵循Python內建的`set`数据结构的运算法则，
        因此并集、交集、差集和其他的集合操作也可以按照熟悉的方式进行
        '''
        ind_A = pd.Index([1,11,41,45,124])
        ind_B = pd.Index([11,24,234,56,1])

        ind_A & ind_B
        ind_A | ind_B
        ind_A ^ ind_B # 互斥差集

        '''
        Python中删除列表元素
        '''
        # 创建一个列表
        list = [1, 2, 3, 4, 5]

        # 使用del关键字删除元素
        del list[1]
        print(list)  # 输出：[1, 3, 4, 5]

        # 使用remove()方法删除元素
        list.remove(3)
        print(list)  # 输出：[1, 4, 5]

        # 使用pop()方法删除元素
        list.pop(0)
        print(list)  # 输出：[4, 5]

        '''
        `Series`对象在很多方面都表现的像一个一维NumPy数组，
        也同时在很多方面表现像是一个标准的Python字典。
        如果我们能将这两个基本概念记住，
        它们能帮助我们理解Series的数据索引和选择的方法
        '''
        import pandas as pd
        import numpy as np
        
        data = pd.Series(np.random.randint(5,size=4),index=['a','b','c','d'])
        data['b']
        '''
        还可以使用标准Python字典的表达式和方法来检查Series的关键字和值
        '''
        'a' in data
        data.keys()
        list(data.items())
        '''
        `Series`对象还可以使用字典操作进行修改。
        就像你可以给字典的一个新的关键字赋值一样，
        你可以新增一个index关键字来扩展`Series`
        '''
        data['e'] = 100
        '''
        `Series`对象构建在字典一样的接口之上，
        并且提供了和NumPy数组一样的数据选择方式，即*切片*，*遮盖*和*高级索引*
        '''
        data['a':'e']
        data[0:2]
        data[(data >= 0) & (data < 4)]
        data[['a','e']]
        '''
        使用指定的显式索引进行切片（例如`data['a':'c']`），
        结束位置的索引值是*包含*在切片里面的，然而，
        使用隐式索引进行切片（例如`data[0:2]`），
        结束位置的索引值是*不包含*在切片里面的
        '''
        data = pd.Series(['a','b','c'],index=[2,66,9])
        data[66]
        data[0:2]
        '''
        因为存在上面看到的这种混乱，
        Pandas提供了一些特殊的*索引符*属性来明确指定使用哪种索引规则。
        这些索引符不是函数，而是用来访问`Series`数据的切片属性
        '''
        '''
        `loc`属性允许用户永远使用显式索引来进行定位和切片
        '''
        data.loc[2]
        data.loc[2:9]
        '''
        `iloc`属性允许用户永远使用隐式索引来定位和切片
        '''
        data.iloc[1]
        data.iloc[0:2]
        '''
        `loc`和`iloc`属性的明确含义使得它们对于维护干净和可读的代码方面非常有效
        '''
        area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
        pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
        data = pd.DataFrame({'area':area, 'pop':pop})
        data['area']
        '''
        当列的名字是字符串时，我们也可以使用属性的方式访问
        '''
        data.area
        data.area is data['area']
        '''
        请记住属性表达式并不是通用的。例如，如果列名不是字符串，
        或者与`DataFrame`的方法名字发生冲突，属性表达式都没法使用。
        例如，`DataFrame`有`pop()`方法，因此，
        `data.pop`将会指向该方法而不是`"pop"`列
        '''
        data.pop is data['pop']
        '''
        应该避免使用属性表达式给列赋值（例如，应该使用`data['pop']=z`而不是`data.pop=z`）
        '''
        data['density'] = data['pop'] / data['area']
        '''
        我们也可以将`DataFrame`看成是一个扩展的二维数组。
        我们可以通过`values`属性查看`DataFrame`对象的底层数组
        '''
        data.values[0]
        '''
        矩阵的倒置
        '''
        data.T
        data['area']
        '''
        Pandas仍然使用`loc`、`iloc`和`ix`索引符来进行操作。
        当你使用`iloc`时，这就是使用隐式索引，
        Pandas会把`DataFrame`当成底层的NumPy数组来处理，
        但行和列的索引值还是会保留在结果中
        '''
        data.iloc[1:3,0:2]
        data.iloc[:3,-1]
        '''
        类似的，使用`loc`索引符时，我们使用的是明确指定的显示索引
        '''
        data.loc[:'Illinois', 'area':'pop']
        '''
        任何NumPy中熟悉的操作都可以在上面的索引符中使用。
        例如，`loc`索引符中我们可以结合遮盖和高级索引模式
        '''
        data.loc[data['density'] > 100, 'area':'pop']
        data.iloc[0,2] = 90
        '''
        还有一些额外的索引规则在实践中也很有用处。首先*索引*是针对列的，
        而切片是针对行的
        '''
        data['Florida':'Illinois']
        data[1:3]
        '''
        直接的遮盖操作也是对行的操作而不是对列的操作
        '''
        data[data['density'] > 100]
        '''
        上面两个规则与NumPy数组语法保持一致，然而他们和Pandas风格可能并不完全一致
        '''
        # 7.1
        '''
        Pandas包括一些NumPy不具备的特性：对于一元运算如取负和三角函数，
        这些ufuncs会在结果中*保留原来的index和column标签*；
        对于二元运算如加法和乘法，Pandas会自动在结果中对参与运算的数据集
        进行*索引对齐*操作
        '''
        '''
        因为Pandas是设计和NumPy一起使用的，因此所有的NumPy通用函数都可以
        在Pandas的`Series`和`DataFrame`对象上使用
        '''
        import numpy as np
        import pandas as pd

        rng = np.random.RandomState(44)
        ser = pd.Series(rng.randint(0,10,4),index=['A','B','C','D'])

        df = pd.DataFrame(rng.randint(0,10,(3,4))
                          ,columns=['A','B','C','D'])
        '''
        如果我们对上面的一个对象使用一元ufunc运算，结果会产生另一个Pandas对象，
        且*保留了索引*
        '''
        np.exp(ser)
        np.sin(df * np.pi / 4)

        area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
        population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
        population / area
        population.index | area.index
        '''
        两个任意输入数据集中对应的另一个数据集不存在的元素都会被设置为`NaN`
        （非数字的缩写），也就是Pandas标示缺失数据的方法。索引的对齐方式会应用在
        任何Python內建的算术运算上，任何缺失的值都会被填充成NaN
        '''
        A = pd.Series([2, 4, 6], index=[0, 1, 2])
        B = pd.Series([1, 3, 5], index=[1, 2, 3])
        A + B
        '''
        如果填充成NaN值不是你需要的结果，你可以使用相应的ufunc函数来计算，
        然后在函数中设置相应的填充值参数
        '''
        A.add(B,fill_value=0)
        '''
        类似的对齐方式在对`DataFrame`操作当中会同时发生在列和行上
        '''
        A = pd.DataFrame(rng.randint(0,20,(2,2)),columns=list('AB'))
        B = pd.DataFrame(rng.randint(0,10,(3,3)),columns=list('BAC')) 
        A + B
        '''
        注意不管索引在输入数据集中的顺序并不会影响结果当中索引的对齐情况
        '''
        '''
        df.stack() 是Pandas库中的一个函数，用于将DataFrame的列“压缩”到索引中。
        这个操作会返回一个新的Series,其中原始DataFrame的列名成为了MultiIndex
        '''
        fill = A.stack().mean() #计算`A`中所有值的平均值
        A.add(B,fill_value=fill)
        '''
        下面列出了Python的运算操作及其对应的Pandas方法：
        | Python运算符 | Pandas方法                      |
        |-----------------|---------------------------------------|
        | ``+``           | ``add()``                             |
        | ``-``           | ``sub()``, ``subtract()``             |
        | ``*``           | ``mul()``, ``multiply()``             |
        | ``/``           | ``truediv()``, ``div()``, ``divide()``|
        | ``//``          | ``floordiv()``                        |
        | ``%``           | ``mod()``                             |
        | ``**``          | ``pow()``                             |
        '''
        '''
        当在`DataFrame`和`Series`之间进行运算操作时，行和列的标签对齐机制依然有效。
        `DataFrame`和`Series`之间的操作类似于在一维数组和二维数组之间进行操作
        '''
        '''
        依据NumPy的广播规则
        '''
        A = rng.randint(10,size=(3,4))
        A - A[1]
        '''
        Pandas中，默认也是采用这种广播机制
        '''
        df = pd.DataFrame(A, columns=list('QRST'))
        df - df.iloc[1]
        df.loc[1:2]
        '''
        如果你希望能够按照列进行减法，你需要使用对应的ufunc函数，然后指定`axis`参数
        '''
        df.subtract(df['R'], axis=0)
        halfrow = df.iloc[0,::2]
        df - halfrow
        '''
        本节介绍的行与列索引保留和对齐机制说明Pandas在进行数据操作时会保持数据的上下文信息，
        因此可以避免同样情况下，使用NumPy数组操作不同形状和异构数据时会发生的错误
        '''
        '''
        8
        UDDDUDUU
        [1,-1,-1,-1,1,-1,1,1]
        [1,0,-1,-2,-1,-2,-1,0]
        _/\      _  
        \    /
            \/\/
        '''
        def countingValleys(steps, path):
            l = []
            count = 0
            for i in path:
                if i == 'U':
                    l.append(1)
                else:
                    l.append(-1)
            
            for j, num in enumerate(l):
                if j > 0:
                    l[j] = l[j-1] + num
            
            for m, mum in enumerate(l):
                if (mum == 0) & (l[m-1] < 0):
                    count += 1
            return count
        '''
        真实的数据很少是干净和同质的。更寻常的情况是，
        很多有意思的数据集都有很多的数据缺失。更复杂的是，
        不同的数据源可能有着不同指代缺失数据的方式
        '''
        '''
        在哨兵值的情况下，哨兵值是某种数据特定的约定值，
        例如用-9999标示一个缺失的整数或者其他罕见的数值，
        又或者使用更加通用的方式，
        比方说标示一个缺失的浮点数为NaN（非数字），
        NaN是IEEE浮点数标准中的一部分。
        Pandas选择了最后一种方案，即通用哨兵值标示缺失值。
        更进一步说就是，使用两个已经存在的Python空值：
        `NaN`代表特殊的浮点数值和Python的`None`对象
        '''
        import pandas as pd
        import numpy as np

        vals1 = np.array([1,None,3,4]) 
        '''
        第一个被Pandas使用的缺失哨兵值是`None`，
        它是一个Python的单例对象，很多情况下它都作为Python代码中
        缺失值的标志。因为这是一个Python对象，`None`不能在任意的
        NumPy或Pandas数组中使用，它只能在数组的数据类型
        是`object`的情况下使用（例如，Python对象组成的数组）
        '''
        '''
        `dtype=object`表示这个NumPy数组的元素类型是Python的对象
        '''
        for dtype in [object,int]:
            print('dtype=',dtype)
            %timeit np.arange(2E6,dtype=dtype).sum()
            print()
        '''
        会比NumPy其他基础类型进行的快速操作消耗更多的执行时间
        '''
        vals1.sum()
        '''
        使用Python对象作为数组数据类型的话，当使用聚合操作如`sum()`
        或`min()`的时候，如果碰到了`None`值，那就会产生一个错误：
        整数和`None`对象之间进行加法运算是未定义的
        '''
        vals2 = np.array([1,np.nan,3,4]) 
        vals2.dtype
        '''
        NumPy使用原始的浮点类型来存储这个数组：
        这意味着不像前面的对象数组，
        这个数组支持使用编译代码来进行快速运算。
        你应该了解到`NaN`就像一个数据的病毒，
        它会传染到任何接触到的数据。不论运算是哪种类型，
        `NaN`参与的算术运算的结果都会是另一个`NaN`
        '''
        1 + np.nan
        0 * np.nan
        '''
        对于这个数组进行的聚合操作是良好定义的（意思是不会发生错误），
        但是却并不十分有意义
        '''
        print(vals2.sum(),vals2.max(),vals2.min())
        '''
        NumPy还提供了一些特殊的聚合函数可以用来忽略这些缺失值
        '''
        import math
        print(np.nansum(vals2),np.nanmax(vals2),math.floor(np.nanmean(vals2)))
        '''
        请记住`NaN`是一个特殊的浮点数值；对于整数、字符串或者其他类型来说都没有对应的值
        '''
        m = pd.Series([1,np.nan,3,None])
        '''
        `NaN`和`None`在Pandas都可以使用，而且Pandas基本上将两者进行等同处理，
        可以在合适的情况下互相转换
        '''
        n = pd.Series(range(2),dtype=int)
        n[0] = None
        '''
        对于哪些没有通用哨兵值的类型，Pandas在发现出现了NA值的情况下会自动对它们进行类型转换。
        例如，如果我们在一个整数数组中设置了一个`np.nan`值，整个数组会自动向上扩展为浮点类型
        '''
        '''
        下表列出了Pandas在出现NA值的时候向上类型扩展的规则：

        |大类型     | 当NA值存在时转换规则 | NA哨兵值      |
        |--------------|-----------------------------|------------------------|
        | ``浮点数`` | 保持不变                   | ``np.nan``             |
        | ``object``   | 保持不变                   | ``None`` 或 ``np.nan`` |
        | ``整数``  | 转换为``float64``         | ``np.nan``             |
        | ``布尔``  | 转换为``object``          | ``None`` 或 ``np.nan`` |
        '''

        data = pd.Series([1,np.nan,'Hello',None])
        data.isnull()
        '''
        - `isnull()`：生成一个布尔遮盖数组指示缺失值的位置
        - `notnull()`：`isnull()`相反方法
        '''
        data[data.notnull()]
        '''
        布尔遮盖数组可以直接在`Series`或`DataFrame`对象上作为索引使用
        '''
        data.dropna()

        df = pd.DataFrame([[1,2,np.nan]
                           ,[2,3,5]
                            ,[np.nan,5,6]])
        '''
        我们不能在`DataFrame`中移除单个空值；
        我们只能移除整行或者整列
        默认，`dropna()`会移除出现了空值的整行
        '''
        df.dropna()
        df.dropna(axis='columns')
        '''
        希望移除那些*全部*是NA值或者大部分是NA值的行或列。
        这可以通过设置`how`或`thresh`参数来实现，
        它们可以更加精细地控制移除的行或列包含的空值个数
        '''
        '''
        默认的情况是`how='any'`，
        因此任何行或列只要含有空值都会被移除。
        你可以将它设置为`how=all`，
        这样只有那些行或列*全部*由空值构成的情况下才会被移除
        '''
        df[3] = np.nan
        df.dropna(axis=1,how='all')
        '''
        `thresh`参数可以让你指定结果中每行或列至少包含
        非空值的个数
        '''
        df.dropna(axis='index',thresh=3) # 行中如果有3个或以上的非空值，将会被保留
        data = pd.Series([1,np.nan,2,None,3],index=list('abcde'))
        data.fillna(0)
        '''
        可以指定填充的方法，如向前填充，将前一个值传播到下一个空值
        '''
        data.fillna(method='ffill')
        '''
        向后填充，使用后一个有效值传播到前一个空值
        '''
        data.fillna(method='bfill')

        df.fillna(axis='columns', method='ffill') # 按列进行向前填充
        '''
        如果空值的前面没有值（此处的`df.loc[2, 0]`前面已经没有列，
        沿着列填充），那么NA值将会保留下来
        '''
        df.fillna(axis=0, method='bfill',inplace=True)

        # 7.3
        import pandas as pd
        import numpy as np

        pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        
        index = [('California', 2000), ('California', 2010),
                 ('New York', 2000), ('New York', 2010),
                 ('Texas', 2000), ('Texas', 2010)]
        populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
        pop = pd.Series(populations,index=index)

        pop[('California', 2000):('New York', 2010)]
        pop.index
        pop[[('New York', 2000),('Texas', 2010)]]
        
        pop[[i for i in pop.index if i[1] == 2000]]
        '''
        Pandas`MultiIndex`类型提供了我们需要的真正的多重索引功能
        '''
        '''
        按照下面的方式从元组创建一个多重索引
        '''
        index = pd.MultiIndex.from_tuples(index)
        '''
        如果我们使用这个`MultiIndex`对我们的series进行重新索引，
        我们可以看到这个数据集的层级展示
        '''
        pop = pop.reindex(index)
        '''
        在多重索引展示中，缺失的索引值数据表示它与上一行具有相同
        的值
        '''
        '''
        Series索引针对行，Dataframe索引针对列
        '''
        s = pd.Series(([[1,2,3],[2,3,4],[4,6,8]]),index=pd.MultiIndex.from_tuples([(1,'a'),(2,'b'),(3,'b')]))
        s[:,'b']
        '''
        使用这个`MultiIndex`对我们的series进行重新索引，
        现在想要获取第二个索引值为2010年的数据，
        只需要简单的使用Pandas的切片语法即可
        '''
        pop[:,2000]
        '''
        Pandas已经内建了这种等同的机制。
        `unstack()`方法可以很快地将多重索引的`Series`转换成普通索引
        的`DataFrame`
        '''
        pop_df = pop.unstack()
        '''
        `stack()`方法提供了相反的操作
        '''
        pop_df.stack()
        # 7.8
        '''
        每个多重索引中的额外层次都代表着数据中额外的维度；
        利用这点我们可以灵活地详细地展示我们的数据，
        例如我们希望在上面各州各年人口数据的基础上增加一列
        （比方说18岁以下人口数）；使用`MultiIndex`能很简单的为`DataFrame`
        增加一列
        '''
        pop_df = pd.DataFrame({'Total':pop,
                               'under18':[9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
        '''
        所有在[在Pandas中操作数据]中介绍过的ufuncs和其他功能也可以应用到层次化索引数据上
        '''
        f_u18 = pop_df['under18'] / pop_df['Total']
        f_u18.unstack()
        '''
        允许我们能简单和迅速的操作数据，甚至是高维度的数据
        '''
        '''
        最直接的构建多重索引`Series`或`DataFrame`的方式是向index参数传递一个多重列表
        np.random.rand(d0, d1, ..., dn) 可以生成一个给定形状的数组，数组中的元素服从 [0, 1) 之间的均匀分布
        '''
        df = pd.DataFrame(np.random.rand(4,2)
                          ,index=[['a','a','b','b'],[1,1,0,0]]
                          ,columns=['data1','data2'])
        '''
        创建`MultiIndex`的工作会自动完成
        如果你使用元组作为关键字的字典数据传给Series，
        Pandas也会自动识别并默认使用`MultiIndex`
        '''
        data = {('California', 2000): 33871648,
            ('California', 2010): 37253956,
            ('Texas', 2000): 20851820,
            ('Texas', 2010): 25145561,
            ('New York', 2000): 18976457,
            ('New York', 2010): 19378102}
        pd.Series(data)
        '''
        有时候显式地创建`MultiIndex`对象也是很有用的
        '''
        '''
        当你需要更灵活地构建多重索引时，你可以使用`pd.MultiIndex`的构造器。
        例如，你可以使用多重列表来构造一个和前面一样的`MultiIndex`对象
        '''
        pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
        pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
        '''
        用两个单一索引的笛卡尔乘积来构造
        '''
        pd.MultiIndex.from_product([['a','b'],[1,2]])
        '''
        你可以用`MultiIndex`构造器来构造多重索引，你需要传递`levels`
        （多重列表包括每个层次的索引值）和`labels`
        （多重列表包括数据点的标签值）参数
        '''
        pd.MultiIndex(levels=[['a','b'],[1,2]],codes=[[1,0],[0,1]])
        '''
        Given an array of integers, find the longest subarray where the absolute difference between any two elements is less than or equal to 1.
        '''
        def pickingNumbers(a):
            longest_ = 0
    
            for i in list(set(a)):
                curr = sum(map(lambda x: 1, filter(lambda x: i == x, a)))
                below = sum(map(lambda x: 1, filter(lambda x: (i - 1) == x, a)))
                above = sum(map(lambda x: 1, filter(lambda x: (i + 1) == x, a)))
                longest_ = max(curr + below, curr + above, longest_)
            return longest_
        '''
        对于每个元素num,计算以下三个值：
            same:在列表a中等于num的元素个数。
            below:在列表a中等于num-1的元素个数。
            above:在列表a中等于num+1的元素个数。
        更新max_subset的值，使其等于当前的max_subset、same+below和same+above三者中的较大值
        '''
        '''
        lambda x: 1:这是一个匿名函数，接受一个参数x,并返回1。
        filter(lambda x: x==num, a):这是Python内置的filter函数，它接受两个参数。
        第一个参数是一个函数，用于测试序列中的每个元素是否满足条件；
        第二个参数是一个可迭代对象(如列表、元组等)。filter函数会返回一个迭代器，
        其中包含满足条件的元素
        '''
        '''
        map()是Python中的一个内置函数，用于将一个函数应用于一个可迭代对象的所有元素，如列表、元组等
        map(lambda x: 1, ...)将使用另一个匿名函数lambda x: 1对筛选后的列表进行映射操作，将每个元素都替换为整数1
        '''
        '''
        set() 是 Python 中的一个内置函数，用于创建一个无序且不重复的元素集合
        '''
        # 7.9
        pd.MultiIndex(levels=[['a','b'],[1,2]],codes=[[1,1,0,0],[0,1,0,1]])
        '''
        levels和codes是为了确定多级索引的对应关系。
        levels的第一级列表[‘a’,‘b’]，对应codes的[0,0,1,1]，codes中是0代表’a’的位置，1代表‘b’的位置。
        leves的第二级列表[1,2]，对应codes的[0,1,0,1]，codes中的0代表1的位置，1代表2的位置。
        两级位置确定后，根据codes位置可绘制标识
        '''
        '''
        这些对象都能作为`index`参数传递给`Series`或`DataFrame`构造器使用，
        或者作为`reindex`方法的参数提供给`Series`或`DataFrame`对象进行重新索引
        '''
        '''
        给`MultiIndex`的不同层次进行命名。这可以通过在上面的`MultiIndex`构造方法中
        传递`names`参数，或者创建了之后通过设置`names`属性来实现
        '''
        pop.index.names = ['state','year']
        '''
        在复杂的数据集中，这种命名方式让不同的索引值保持它们原本的意义
        '''
        '''
        在一个`DataFrame`中，行和列是完全对称的，就像前面看到的行可以有多层次的索引，
        列也可以有多层次的索引
        '''
        index = pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])
        columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],names=['subject','type'])
        
        data = np.round(np.random.randn(4,6),1)
        data[:, ::2] *= 10
        data += 37

        health_data = pd.DataFrame(data,index=index,columns=columns)
        '''
        上面的数据集实际上是一个四维的数据，四个维度分别是受试者、测试类型、
        年份和测试编号。创建了这个`DataFrame`之后，
        我们可以使用受试者的姓名来很方便的获取到此人的所有测试数据
        '''
        health_data['Bob']
        '''
        对于这种包含着多重标签的多种维度（人、国家、城市等）数据。
        使用这种层次化的行和列的结构会非常方便
        '''
        '''
        在`MultiIndex`上进行检索和切片设计的非常直观，
        你可以将其想象为在新增的维度上进行检索
        '''
        pop['California',2000]
        '''
        即仅在索引中检索其中的一个层次。得到的结果是另一个`Series`但是
        具有更少的层次结构
        '''
        pop['California']
        '''
        部分切片同样也是支持的，只要`MultiIndex`是排序的
        '''
        pop['California':'Texas']
        '''
        在有序索引的情况下，部分检索也可以用到低层次的索引上，
        只需要在第一个索引位置传递一个空的切片即可
        '''
        pop[:,2010]
        pop[pop > 22000000]
        pop[['California','Texas']]
        '''
        请注意`DataFrame`中主要的索引是列
        '''
        health_data['Guido','HR']
        health_data.iloc[:2,:2]
        health_data.loc[:,('Bob','HR')]
        '''
        使用这种索引元组并不是特别的方便；例如试图在元组中使用切片会产生一个语法错误
        '''
        # health_data.loc[(:, 1), (:, 'HR')]
        '''
        解决上述问题的方法有一个更好的方式是使用`IndexSlice`对象，
        该对象是Pandas专门为这种情况准备的
        '''
        idx = pd.IndexSlice
        health_data.loc[idx[:,1],idx[:,'HR']]
        '''
        访问多重索引的`Series`和`DataFrame`对象中的数据有很多种方法，
        除了阅读本书中介绍的这些工具外，熟悉它们的最好方式就是在实践中使用它们
        '''
        '''
        这里我们要强调一下。*如果索引是无序的话，很多`MultiIndex`的切片操作都会失败*
        '''
        index = pd.MultiIndex.from_product([['a','c','b'],[1,2]])
        data = pd.Series(np.random.rand(6),index=index)
        data.index.names = ['char','int']

        try:
            data['a':'b']
        except KeyError as e:
            print(type(e))
            print(e)
        '''
        这是MultiIndex没有排序的结果。许多因素决定了，当对`MultiIndex`进行部分
        的切片和其他相似的操作时，都需要索引是有序（或者说具有自然顺序）的。
        Pandas提供了方法来对索引进行排序；例如`DataFrame`对象的`sort_index()`
        和`sortlevel()`方法。我们在这里使用最简单的`sort_index()`方法
        '''
        data = data.sort_index()
        data['a':'b']
        '''
        将一个堆叠的多重索引的数据集拆分成一个简单的二维形式，
        还可以指定使用哪个层次进行拆分
        '''
        pop.unstack(level=0)
        pop.unstack(level=1)
        '''
        `unstack()`的逆操作是`stack()`，我们可以使用它来重新堆叠数据集
        '''
        pop.unstack().stack()
        '''
        还有一种重新排列层次化数据的方式是将行索引标签转为列索引标签；
        这可以使用`reset_index`方法来实现,为了清晰起见，我们可以设置列的标签
        '''
        pop_flat = pop.reset_index(name='population')
        '''
        通常当我们处理真实世界的数据的时候，我们看到的就会是如上的数据集的形式，
        因此从列当中构建一个`MultiIndex`会很有用。
        这可以通过在`DataFrame`上使用`set_index`方法来实现，
        这样会返回一个多重索引的`DataFrame`
        '''
        pop_flat.set_index(['state','year'])
        '''
        前面我们已经了解到Pandas有內建的数据聚合方法，例如`mean()`、`sum()`和
        `max()`。对于层次化索引的数据而言，这可以通过传递`level`参数来控制数据
        沿着哪个层次的索引来进行计算
        '''
        '''
        将每年测量值进行平均。我们可以用level参数指定我们需要进行聚合的标签，
        这里是年份
        '''
        data_mean = health_data.mean(level='year')
        data_mean.mean(axis=1,level='type')
        '''
        虽然只有两行代码，我们已经能够计算得到所有受试者每年多次测试取样的平均的心率和提问。
        这个语法实际上是`GroupBy`函数的一种简略写法，我们会在[聚合和分组]一节中详细介绍。
        虽然这只是一个模拟的数据集，但是很多真实世界的数据集也有相似的层次化结构
        '''
        '''
        Panel结构，因为作者认为在大多数情况下多重索引会更加有用，在表现高维数据时概念也会显得更加简单。
        而且更加重要的是，面板数据从基本上来说是密集数据，而多重索引从基本上来说是稀疏数据。
        随着维度数量的增加，使用密集数据方式表示真实世界的数据是非常的低效的。
        但是对于一些特殊的应用来说，这些结构是很有用的。
        '''

        # 7.10
        '''
        很多对数据进行的有趣的研究都来源自不同数据源的组合。
        这些组合操作包括很直接的连接两个不同的数据集，
        到更复杂的数据库风格的联表和组合可以正确的处理数据集之间的重复部分。
        `Series`和`DataFrame`內建了对这些操作的支持，
        Pandas提供的函数和方法能够让这种数据操作高效而直接。
        '''
        import numpy as np
        import pandas as pd

        def make_df(cols, ind):
            data = {c: [str(c) + str(n) for n in ind] for c in cols}
            return pd.DataFrame(data,index=ind)
        
        make_df('ABC',range(3))

        dic_ = {1: 'A', 2: 'B'}
        index = [0]
        dic__ = {'A': 1, 'B': 2}
        pd.DataFrame(dic_,index=index)

        class display(object):
            """多个对象的HTML格式展示"""
            template = """<div style="float: left; padding: 10px;">
            <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
            </div>"""
            def __init__(self, *args):
                self.args = args
                
            def _repr_html_(self):
                return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                                    for a in self.args)

            def __repr__(self):
                return '\n\n'.join(a + '\n' + repr(eval(a))
                                    for a in self.args)
        '''
        `Series`和`DataFrame`对象的连接与NumPy数组的连接非常相似，
        NumPy数组我们可以通过`np.concatenate`函数来实现
        '''
        x = [1,2,3]
        y = [4,5,6]
        z = [7,8,9]
        np.concatenate((x,y,z))
        '''
        第一个参数是需要进行连接的数组的元组或列表。
        函数还可以提供一个`axis`关键字参数来指定沿着哪个维度方向对数组进行连接
        '''
        x = [[1,2]
             ,[3,4]]
        np.concatenate((x,x),axis=0)
        '''
        Pandas有相应的函数`pd.concat()`，与`np.concatenate`有着相似的语法，
        但是有一些参数我们需要深入讨论：
        ```python
        # Pandas v0.24.2的函数签名
        pd.concat(
            objs,
            axis=0,
            join='outer',
            join_axes=None,
            ignore_index=False,
            keys=None,
            levels=None,
            names=None,
            verify_integrity=False,
            sort=None,
            copy=True,
        )
        '''
        '''
        `pd.concat()`可以用来对`Series`或`DataFrame`对象进行简单的连接，
        就像可以用`np.concatenate()`来对数组进行简单连接一样
        '''
        sers_1 = pd.Series(['A','B','C'], index=[1,2,3])
        sers_2 = pd.Series(['D','E','F'], index=[4,5,6])
        pd.concat([sers_1,sers_2])

        df_1 = make_df('AB', [1,2])
        df_2 = make_df('BC', [3,4])
        display('df_1','df_2','pd.concat((df_1,df_2))')
        '''
        默认情况下，连接会按照`DataFrame`的行来进行（即`axis=0`）。
        就像`np.concatenate`那样，`pd.concat`允许指定沿着哪个维度方向进行连接
        '''
        df_3 = make_df('AB', [0,1])
        df_4 = make_df('CD', [0,1])
        display('df_3','df_4',"pd.concat((df_3,df_4),axis='columns')")
        '''
        也可以使用相同的声明方式`axis=1`；这里我们使用了更加直观的方式`axis='columns'`
        '''
        '''
        `np.contenate`和`pd.concat`的一个重要区别是Pandas的连接会*保留行索引*，
        甚至在结果中包含重复索引的情况下
        '''
        df_1 = make_df('AB', [1,2])
        df_2 = make_df('AB', [3,4])
        df_2.index = df_1.index
        display('df_1', 'df_2', 'pd.concat((df_1,df_2))')
        '''
        看到结果中的重复索引。虽然这是`DataFrame`允许的，
        但是结果通常不是你希望的。`pd.concat()`提供了一些处理这个问题的方法
        '''
        '''
        验证`pd.concat()`结果数据集中是否含有重复的索引，
        你可以传递参数`verify_integrity=True`参数。
        这时连接结果的数据集中如果存在重复的行索引，将会抛出一个错误
        '''
        try:
            pd.concat([df_1,df_2],verify_integrity=True)
        except ValueError as e:
            print(e)
        '''
        有些情况下，索引本身并不重要，那么可以选择忽略它们。
        给函数传递一个`ignore_index=True`的参数，
        `pd.concat`函数会忽略连接时的行索引，
        并在结果中重新创建一个整数的索引值
        '''
        display('df_1','df_2','pd.concat([df_1,df_2],ignore_index=True)')
        '''
        还有一种方法是使用`keys`参数来指定不同数据集的索引标签；
        这时`pd.concat`的结果会是包含着连接数据集的多重索引数据集
        '''
        df = pd.concat([df_1,df_2],keys=['df_1','df_2'])
        df[['B','A']]
        flat_df = df.unstack(level=0)
        '''
        在实际情况中，从不同源得到的数据通常具有不同的列数或者列标签，
        `pd.concat`提供了几个相应的参数帮助我们完成上面的任务。
        下例中的两个数据集只有部分（非全部）列和标签相同
        '''
        df5 = make_df('ABC', [1, 2])
        df6 = make_df('BCD', [3, 4])
        display('df5', 'df6', 'pd.concat([df5, df6])')
        '''
        默认情况下，那些对应源数据集中不存在的元素值，将被填充为NA值。
        如果想改变默认行为，我们可以通过指定`join`和`join_axes`参数来实现。
        `join`参数默认为`join='outer'`，就像我们上面看到的情况，结果是数据集的并集；
        如果将`join='inner'`传递给`pd.concat`，
        那么就会是数据源中相同的列保留在结果中，因此结果是数据集的交集
        '''
        display('df5','df6', "pd.concat([df5, df6], join='inner')")
        '''
        可以通过另一个方法`reindex`来指定结果中保留的列，
        该参数接受被保留索引标签的列表。
        下例中我们指定结果中的列和第一个进行连接的数据集完全相同
        '''
        '''
        可以通过`reindex`方法达到同样的目的，下面使用了`reindex`语法保留了`df5`的所有列
        '''
        display('df5','df6', 'pd.concat([df5, df6]).reindex(df5.columns,axis=1)')
        '''
        `pd.concat`函数的参数很多，组合使用它们能解决组合多个数据集中的很多问题；
        请记住当你在自己的数据上操作时，你可以灵活地应用它们，完成你的工作目标
        '''
        '''
        `Series`和`DataFrame`对象都有一个`append`方法，
        它能完成和`pd.concat`一样的功能，并能让让你写代码时节省几次敲击键盘的动作
        '''
        display('df_1', 'df_2', 'df_1.append(df_2)')
        '''
        Pandas中的`append()`方法不会修改原始参与运算的数据集，有需要连接多个数据集时，
        应该避免多次使用`append`方法，而是将所有需要进行连接的数据集形成一个列表，
        并传递给`concat`函数来进行连接操作
        '''
        # 一些易犯错误
        import pandas as pd
        import numpy as np

        # data = [(1,2),(3,4)]
        data = {1: 'a', 2: 'b', 1: 'c', 2: 'd'}
        index = range(2)
        df = pd.DataFrame(data,index=index)
        '''
        有时建立df时需要指定index，不然会报If using all scalar values,
        you must pass an index
        '''
        data.items()
        '''
        dict.items() 是 Python 字典对象的一个方法，
        它返回一个包含字典中所有键值对的视图对象。每个元素都是一个元组，
        其中第一个元素是键，第二个元素是对应的值
        '''
        '''
        这个错误是因为在使用pd.DataFrame()创建DataFrame时，
        传入的字典中所有值都是标量(即单个数值),但是没有提供索引。
        要解决这个问题，可以在创建DataFrame时传入一个索引
        '''
        '''
        这个错误是因为在创建pandas的Index对象时，传入了一个字符串'ABC',
        而不是一个集合类型。要解决这个问题，
        需要确保传入的数据是一个集合类型，例如列表、元组或数组等
        '''
        '''
        An arcade game player wants to climb to the top of the leaderboard and track their ranking.
        The game uses Dense Ranking. so itsleaderboard works like this:
            The player with the highest score is ranked number 1 on the leaderboard.
            Plavers who have equal scores receive the same ranking number, 
            and the next player(s) receive the immediately folowing rankingnumber.
        
        Example:
            ranked =[100,90,90,80]
            player =「70,80,105]
            The ranked players will have ranks 1.2. 2. and 3, respectively. 
            lf the player's scores are 70, 80 and 105. their rankings after each gameare 4th, 
            3rd and 1st . Retum [4, 3, 1].

        The existing leaderboard, ranked, is in descending order.
        The player's scores, player, are in ascending order.
        '''
        def climbingLeaderboard(ranked, player): 
            ranked = sorted(list(set(ranked)),reverse=True)

            l = len(ranked)
            result = []

            for p in player:
                while (l > 0) and (p >= ranked[l - 1]):
                    l -= 1
                
                result.append(l + 1)
            
            return result
        # 7.17
        '''
        Pandas提供的一个基本的特性就是它的高性能、内存中进行的联表和组合操作。
        如果你使用过数据库，你应该已经很熟悉相关的数据操作了。
        Pandas在这方面提供的主要接口是`pd.merge`函数
        '''
        import pandas as pd
        import numpy as np

        class display(object):
            """Display HTML representation of multiple objects"""
            template = """<div style="float: left; padding: 10px;">
            <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
            </div>"""
            def __init__(self, *args):
                self.args = args
                
            def _repr_html_(self):
                return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                                for a in self.args)
            
            def __repr__(self):
                return '\n\n'.join(a + '\n' + repr(eval(a))
                                for a in self.args)
        '''
        `pd.merge()`实现的是我们称为*关系代数*的一个子集，
        关系代数是一系列操作关系数据的规则的集合，它构成了大部分数据库的数学基础
        '''
        '''
        `pd.merge()`函数实现了几种不同类型的联表：*一对一*、*多对一*和*多对多*。
        所有三种类型的联表都可以通过`pd.merge()`函数调用来实现；
        具体使用了哪种类型的联表取决于输入数据的格式。
        '''
        '''
        最简单的联表操作类型就是一对一连接，在很多方面，
        这种联表都和我们在[组合数据集：Concat 和 Append]中看到的按列进行数据集
        连接很相似
        '''
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                           'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                           'hire_date': [2004, 2008, 2012, 2014]})
        display('df1','df2')
        df3 = pd.merge(df1,df2)
        '''
        `pd.merge()`函数会自动识别每个`DataFrame`都有"employee"列，
        因此会自动按照这个列作为键对双方进行合并。
        合并的结果通常会丢弃了原本的行索引标签，除非在合并时制定了行索引
        '''
        '''
        多对一联表的情况发生在两个数据集的关键字列上的其中一个含有重复数据的时候。
        在这种多对一的情况下，结果的`DataFrame`会正确的保留那些重复的键值
        '''
        df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                            'supervisor':['Carly', 'Guido', 'Steve']})
        display('df3','df4','pd.merge(df3,df4)')
        '''
        结果的`DataFrame`多了一列`supervisor`，上面的数据也是按照`group`的
        重复情况进行重复的
        '''
        '''
        如果左右的数据集在关键字列上都有重复数据，那么结果就是一个多对多的组合
        '''
        df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                            'skill': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
        display('df1','df5','pd.merge(df1,df5)')
        '''
        在实践中，数据集极少好像我们上面的例子那样干净。`pd.merge()`提供的一些参数，
        能精细的对连接操作进行调整
        '''
        '''
        通常情况下，列名并不会这么匹配，`pd.merge()`提供了一系列的参数来处理这种情况
        '''
        '''
        你可以使用`on`关键字参数明确指定合并使用的关键字列名，
        参数可以是一个列名或者一个列名的列表
        '''
        display('df1','df2',"pd.merge(df1,df2,on='employee')")
        '''
        该参数仅在左右两个`DataFrame`都含有相同的指定列名的情况下有效
        '''
        '''
        使用不同列名来合并两个数据集的情况下；例如，我们有一个数据集，
        在它里面员工姓名的列名不是"employee"而是"name"。在这种情况下，
        我们可以使用`left_on`和`right_on`关键字来分别指定两个列的名字
        '''
        df3 = pd.DataFrame({'name':['Bob', 'Jake', 'Lisa', 'Sue'],
                            'salary':[70000, 80000, 120000, 90000]})
        display('df1','df3',"pd.merge(df1,df3,left_on='employee',right_on='name')")
        '''
        结果中有一个冗余的列，我们可以将该列移除，例如使用`DataFrame`的`drop()`方法
        '''
        pd.merge(df1,df3,left_on='employee',right_on='name').drop('name',axis=1)
        '''
        有时候，你不是需要按列进行合并，而是需要按照行索引进行合并
        '''
        df1a = df1.set_index('employee')
        df2a = df2.set_index('employee')
        display('df1a','df2a')
        '''
        通过指定`left_index`和`right_index`标志参数，你可以将两个数据集
        按照行索引进行合并
        '''
        display('df1a','df2a','pd.merge(df1a,df2a,left_index=True,right_index=True)')
        '''
        为了方便，`DataFrame`实现了`join()`方法，默认按照行索引合并数据集
        '''
        display('df1a','df2a','df1a.join(df2a)')
        '''
        如果需要混合的进行行或列的合并，你可以通过混合指定`left_index`和
        `right_on`参数或者`left_on`和`right_index`参数来实现
        '''
        display('df1a','df3','pd.merge(df1a,df3,left_index=True,right_on="name")')
        '''
        所有上面的参数都能应用到多重行索引和/或多重列上
        '''
        '''
        在上面的例子中，我们都忽略了在进行数据集合并时一个重要的内容：
        合并时所使用的集合算术运算类型。这部分内容对于当一个数据集的键值在
        另一个数据集中不存在时很有意义
        '''
        df6 = pd.DataFrame({'name':['Peter', 'Paul', 'Mary'],
                            'food':['fish', 'beans', 'bread']},
                            columns=['name','food'])
        df7 = pd.DataFrame({'name':['Mary', 'Joseph'],
                           'drink':['wine', 'beer']},
                           columns=['name','drink'])
        display('df6','df7','pd.merge(df6,df7)')
        '''
        上面我们合并的两个数据集在关键字列上只有一个"name"数据是共同的：Mary。
        默认情况下，结果会包含两个集合的*交集*；这被称为*内连接*。
        我们显式的指定`how`关键字参数，它的默认值是`"inner"`
        '''
        pd.merge(df6,df7,how='inner')
        '''
        `how`参数的其他选项包括`'outer'`、`'left'`和`'right'`。
        *外连接outer*会返回两个集合的并集，并将缺失的数据填充为Pandas的NA值
        '''
        pd.merge(df6,df7,how='outer')
        '''
        *左连接left*和*右连接right*返回的结果是包括所有的左边或右边集合
        '''
        pd.merge(df6,df7,how='left')
        '''
        结果中的行与左集合保持一致。使用`how='right'`结果会和右集合保持一致。
        所有这些集合运算类型可以和前面的连接类型组合使用
        '''
        df8 = pd.DataFrame({'name':['Bob', 'Jake', 'Lisa', 'Sue'],
                            'rank':[1, 2, 3, 4]})
        df9 = pd.DataFrame({'name':['Bob', 'Jake', 'Lisa', 'Sue'],
                           'rank':[3, 1, 4, 2]})
        display('df8','df9','pd.merge(df8,df9,on="name")')
        '''
        因为结果可能会有两个相同的列名，发生冲突，merge函数会自动为这两个列
        添加`_x`和`_y`后缀，使得输出结果每个列名称唯一。如果默认的后缀不是
        你希望的，可以使用`suffixes`关键字参数为输出列添加自定义的后缀
        '''
        display('df8','df9','pd.merge(df8,df9,on="name",suffixes=["_L","_R"])')
        '''
        这些后缀可以应用在所有的连接方式中，也可以在多个列冲突时使用
        '''
        '''
        合并及联表操作在你处理多个不同数据来源时会经常出现
        '''

        pop = pd.read_csv(r'F:\git\notebooks\data\state-population.csv')
        areas = pd.read_csv(r'F:\git\notebooks\data\state-areas.csv')
        abbrevs = pd.read_csv(r'F:\git\notebooks\data\state-abbrevs.csv')

        display('pop.head()','areas.head()','abbrevs.head()')
        '''
        需要计算一个相对非常直接的结果：根据美国各州2010年人口密度进行排名
        '''
        merged = pd.merge(pop,abbrevs,left_on='state/region',right_on='abbreviation',
                          how='outer')
        merged.drop('abbreviation',axis=1,inplace=True)
        '''
        让我们检查结果中是否有不匹配的情况，通过在数据集中寻找空值来查看
        '''
        merged.isnull().any()
        merged[merged['population'].isnull()].head()
        '''
        发现所有空的人口数据都是2000年前波多黎各的；
        这可能因为数据来源本来就没有这些数据造成的
        '''
        '''
        更重要的是，我们发现一些新的州`state`的数据也是空的，
        这意味着`abbrevs`列中不存在这些州的简称。
        再看看是哪些州有这种情况
        '''
        merged.loc[merged['state'].isnull(),'state/region'].unique()
        '''
        从上面的结果很容易发现：人口数据集中包括波多黎各（PR）和
        全美国（USA）的数据，而州简称数据集中却没有这两者数据。
        通过填充相应的数据可以很快解决这个问题
        '''
        merged.loc[merged['state/region']=='PR','state'] = 'Puerto Rico'
        merged.loc[merged['state/region']=='USA','state'] = 'United States'
        merged.isnull().any()

        final = pd.merge(merged,areas,on='state',how='left')
        final.head()

        final.isnull().any()

        final.loc[final['area (sq. mi)'].isnull(),'state'].unique()

        final['state'][final['area (sq. mi)'].isnull()].unique()

        final.dropna(inplace=True)
        final.head()

        data2010 = final.query('ages == "total" & year == 2010')
        data2010.head()

        data2010.set_index('state',inplace=True)
        data2010.head()
        density = data2010['population'] / data2010['area (sq. mi)']
        density.head()

        density.sort_values(ascending=False,inplace=True)
        density.head()
        '''
        我们也可以查看结果的最后部分
        '''
        density.tail()
        '''
        当使用真实世界数据回答这种问题的时候，这种数据集的合并是很常见的任务
        '''
        '''
        df.isnull() 返回整个数据框每个元素对应位置的布尔值
        df.isnull().any() 返回整个数据框每一列的布尔值
        df.isnull().any().any() 返回整个数据框的布尔值
        '''
        '''
        df.unique() 是 pandas 库中的一个方法，
        用于返回一个 Series 对象，其中包含了指定列或者多个列中的唯一值
        '''
        '''
        sort_values() 是 pandas 库中的一个方法，
        用于对 DataFrame 或 Series 中的数据进行排序。
        该方法可以根据一个或多个列的值来对数据进行升序或降序排列
        '''
        data2010.sort_values(by='population', ascending=False)
        '''
        reset_index()：这个函数用于重置数据框的索引。它会将原来的索引列转换为普通列，
        并添加一个新的从0开始的整数索引。默认情况下，新索引会放在数据框的最后一列。
        '''
        '''
        set_index()：这个函数用于设置数据框的索引。你可以将一个或多个列设置为索引
        （把列索引转为行索引）
        '''
        '''
        df.loc[row_label, column_label]
        其中，row_label表示行标签，column_label表示列标签
        '''
        '''
        df.query() 是 pandas 库中的一个方法，用于对 DataFrame 进行过滤。
        它接受一个字符串参数，该字符串表示查询条件，然后返回满足条件的行组成的
        新DataFrame
        '''
        '''
        df.sort_values() 是一个用于对 Pandas DataFrame 进行排序的函数。
        它可以根据一个或多个列的值对数据进行升序或降序排序
        参数说明：
        by：
            指定要排序的列名或列名列表，可以是单个列名或多个列名组成的列表。
        axis：
            指定排序的轴向，0 表示按行排序，1 表示按列排序。默认为 0。
        ascending：
            指定排序方式，True 表示升序，False 表示降序。默认为 True。
        inplace：
            指定是否在原始 DataFrame 上进行排序操作。True 表示在原始 DataFrame 上进行排序，
            False 表示返回一个新的排序后的DataFrame。默认为 False。
        kind：
            指定排序算法的类型。可选值包括 'quicksort'（快速排序）、
            'mergesort'（归并排序）和 'heapsort'（堆排序）。默认为 'quicksort'。
        na_position：
            指定缺失值的位置。可选值包括 'first'（放在前面）、'last'（放在后面）和
            'remove'（删除包含缺失值的行）。默认为 'last'。
        '''
        '''
        删除行：df.drop([index_label], axis=0, inplace=False)
        删除列：df.drop([column_label], axis=1, inplace=False)
        '''
        '''
        There is a list of 26 character heights aligned by index to their letters. 
        For example, 'a' is at index 0 and '2' is at index 25There will also be a string. Using the letter heights given, determine the area of the rectangle highlight in mm'assuming all letters are lmm, wide.
        Exampleh=[1,3,1,3,1,4,1,3,2,5,5,5,5,1,1,5,5,1,5,2,5,5,5,5,5,5]word -'torn'
        The heights are t=2,o= 1,, = 1 and n = 1. The tallest letter is 2 high and 
        there are 4 letters. The hightlightedarea will be 2*4=8mm' so the answer is 8.
        '''
        def designerPdfViewer(h, word):
            l = list('abcdefghijklmnopqrstuvwxyz')
            dic = {}
            length = len(h)
            result = []

            for i in range(length):
                dic[l[i]] = h[i]
            
            for j in word:
                result.append(dic.get(j))
            
            return len(word)*max(result)
        '''
        dict.get(key, default=None): 返回字典中指定键的值，如果键不存在，则返回默认值
        dict.pop(key, default=None): 删除并返回字典中指定键的值，如果键不存在，则返回默认值
        dict.setdefault(key, default=None): 如果键存在于字典中，则返回其值；
        否则将添加键并将其设置为默认值，然后返回默认值
        '''
        # 7.22
        import numpy as np
        import pandas as pd

        class display(object):
            """Display HTML representation of multiple objects"""
            template = """<div style="float: left; padding: 10px;">
            <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
            </div>"""
            def __init__(self, *args):
                self.args = args
                
            def _repr_html_(self):
                return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                                for a in self.args)
            
            def __repr__(self):
                return '\n\n'.join(a + '\n' + repr(eval(a))
                                for a in self.args)
        
        import seaborn as sns
        
        planets = sns.load_dataset('planets')
        planets.shape

        planets.head()

        rng = np.random.RandomState(42)
        sns = pd.Series(rng.rand(5))
        sns

        sns.sum()
        sns.mean()
        '''
        对于`DataFrame`来说，默认情况下是每个列进行聚合的结果
        '''
        df = pd.DataFrame({'A': rng.rand(5),
                           'B': rng.rand(5)})
        df.mean()
        '''
        通过指定`axis`参数，可以为每一行进行聚合操作
        '''
        df.mean(axis='columns')
        # 7.23
        '''
        Pandas提供了很方便的`describe()`可以用来对每个列计算这些通用的聚合结果
        '''
        planets.dropna().describe()
        '''
        对于开始理解数据集的整体情况来说，这是一个非常有用的方法。
        例如，在发现年份`year`列上，结果显示，虽然第一颗太阳系外行星是1989年
        发现的，但是一半的行星直到2010年以后才被发现的。
        这多亏了*开普勒Kepler*计划，它是一个太空望远镜，
        专门设计用来寻找其他恒星的椭圆轨道行星的
        '''
        '''
        下表概括了Pandas內建的聚合操作：
        | 聚合函数              | 描述                     |
        |--------------------------|---------------------------------|
        | ``count()``              | 元素个数           |
        | ``first()``, ``last()``  | 第一个和最后一个元素             |
        | ``mean()``, ``median()`` | 平均值和中位数                 |
        | ``min()``, ``max()``     | 最小和最大值             |
        | ``std()``, ``var()``     | 标准差和方差 |
        | ``mad()``                | 平均绝对离差         |
        | ``prod()``               | 所有元素的乘积            |
        | ``sum()``                | 所有元素的总和                |
        它们都是`DataFrame`和`Series`对象的方法
        '''
        '''
        然而要深入了解数据，简单的聚合经常是不够的。
        `groupby`操作为我们提供更高层次的概括功能，
        通过它能很快速和有效地计算子数据集的聚合数据
        '''
        '''
        `groupby`*拆分、应用、组合*
        `groupby`完成的工作：
        - 拆分*split*步骤表示按照指定键上的值对`DataFrame`进行拆分
            和分组的功能。
        - 应用*apply*步骤表示在每个独立的分组上调用某些函数进行计算，
            通常是聚合、转换或过滤。
        - 组合*combine*步骤将上述计算的结果重新合并在一起输出。
        '''
        df = pd.DataFrame({'key':['A','B','C','A','B','C'],
                           'value': range(6)}, columns=['key','value'])
        '''
        最基础的拆分-应用-组合操作可以使用`DataFrame`的`groupby()`方法来实现，
        方法中传递作为键来运算的列名
        '''
        df.groupby('key')
        '''
        它是`DataFrame`对象的一个特殊的视图，
        使用它可以很容易的研究分组的数据，但是除非聚合操作发生，
        否则它不会进行真实的运算。这种“懒运算”的方式意味着通用的聚合可以
        实现得非常的高效，而对用户来说几乎是透明的
        '''
        '''
        要产生结果，我们可以将一个聚合操作应用到该`DataFrameGroupBy`对象上，
        这样就会在分组上执行应用/组合的步骤，并产生需要的结果
        '''
        df.groupby('key').sum()
        '''
        `GroupBy`对象是一个很灵活的抽象。在很多情况下，
        你可以将它简单的看成`DataFrame`的集合，它在底层做了很多复杂的工作
        '''
        '''
        `GroupBy`对象支持列索引，与`DataFrame`相同，
        返回的是修改后的`GroupBy`对象
        '''
        planets.groupby('method')['orbital_period']
        '''
        我们在原始的`DataFrame`中选择了特定的`Series`，
        这个`Series`是按照提供的列名进行分组的。
        `GroupBy`对象在调用聚合操作之前是不会进行计算的
        '''
        planets.groupby('method')['orbital_period'].median()
        '''
        `GroupBy`对象支持在分组上直接进行迭代，
        每次迭代返回分组的一个`Series`或`DataFrame`对象
        '''
        for method, group in planets.groupby('method'):
            print("{0:30s}: shape={1}".format(method,group.shape))
        '''
        这种做法在某些需要手动实现的情况下很有用，
        虽然通常来说使用內建的`apply`函数会快很多
        '''
        '''
        代码通过for循环遍历"planets"数据表中按照'method'列进行分组的结果。
        在每次循环中，变量method和group分别表示当前组的名称
        （即'method'列的值）和对应的数据子集
        '''
        '''
        print("{0:30s}") 是一个格式化字符串的示例。
        在Python中，我们可以使用花括号 {} 来表示占位符，
        其中的数字 0 表示第一个参数，即要插入到字符串中的值。
        冒号 : 后面的 30s 是一个格式说明符，表示将第一个参数转换为一个长度
        为30个字符的字符串。如果该字符串的长度小于30，那么它将用空格填充
        到30个字符。如果该字符串的长度大于30，那么它将被截断以适应30个字符。
        例如，如果我们调用 print("{0:30s}".format("Hello"))，
        那么输出将是 "Hello"，后面跟着27个空格，使得整个字符串的长度为30
        '''
        '''
        通过一些Python面向对象的魔术技巧，
        任何非显式定义在`GroupBy`对象上的方法，
        无论是`DataFrame`还是`Series`对象的，都可以给分组来调用。
        例如，你可以在数据分组上调用`DataFrame`的`describe()`方法，
        对所有分组进行通用的聚合运算
        '''
        planets.groupby('method')['year'].describe()
        '''
        任何正确的`DataFrame`或`Series`方法都能在相应的`GroupBy`对象上使用，
        这种扩展方法的方式提供了非常灵活及强大的操作
        '''
        '''
        `GroupBy`对象有`aggregate()`、`filter()`、`transfrom`和`apply()`方法，
        它们能在组合分组数据之前有效地实现大量有用的操作
        '''
        rng = np.random.RandomState(0)
        df = pd.DataFrame({'key': list('ABCABC'),
                           'data1': range(6),
                           'data2': rng.randint(0,10,6)},
                           columns=['key','data1','data2'])
        '''
        `aggregate()`方法能提供更多的灵活性。它能接受字符串、函数或者一个列表，
        然后一次性计算出所有的聚合结果
        '''
        df.groupby('key').aggregate(['min',np.median,max])
        '''
        可以将一个字典，里面是列名与操作的对应关系，
        传递给`aggregate()`来进行一次性的聚合运算
        '''
        df.groupby('key').aggregate({'data1': 'min',
                                     'data2': 'max'})
        '''
        过滤操作能在分组数据上移除一些你不需要的数据
        可以认为`filter()`类似于SQL中的HAVING
        '''
        def filter_func(x):
            return x['data2'].std() > 4
        
        display('df',"df.groupby('key').std()","df.groupby('key').filter(filter_func)")
        '''
        用来进行过滤的函数必须返回一个布尔值，表示分组是否能够通过过滤条件。
        上例中A分组的标准差不是大于4，因此整个分组在结果中被移除了
        '''
        '''
        聚合返回的是分组简化后的数据集，而转换可以返回完整数据转换后并重新合并的
        数据集。因此转换操作的结果和输入数据集具有相同的形状。
        一个通用例子是将整个数据集通过减去每个分组的平均值进行中心化
        '''
        df.groupby('key').transform(lambda x: x - x.mean())
        df.groupby('key').mean()
        '''
        `apply()`方法能让你将分组的结果应用到任意的函数上。
        该函数必须接受一个`DataFrame`参数，返回一个Pandas对象（如`DataFrame`、`Series`）
        或者一个标量；组合操作会根据返回的类型进行适配
        '''
        def norm_by_data2(x):
            x['data1'] /= x['data2'].sum()
            return x
        
        display('df',"df.groupby('key').apply(norm_by_data2)")
        '''
        `GroupBy`对象的`apply()`方法是非常灵活的：
        唯一的限制就是应用的函数要接受一个`DataFrame`参数并且返回一个Pandas对象或者标量；
        函数体内做什么工作完全是自定义的
        '''
        '''
        用一个列名对`DataFrame`进行拆分。这只是分组的众多方式的其中之一
        '''
        '''
        分组使用的键可以使任何的序列或列表，只要长度和`DataFrame`的长度互相匹配即可
        '''
        l = [0,1,0,1,2,0]
        display('df',"df.groupby(l).sum()")
        '''
        前面的`df.groupby('key')`语法还有另外一种更加有含义的方式来实现
        '''
        display('df',"df.groupby(['key']).sum()")
        '''
        还有一种方法是提供一个字典，将索引值映射成分组键
        '''
        df2 = df.set_index('key')
        mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
        display('df2', "df2.groupby(mapping).sum()")
        '''
        类似映射，你可以传递任何Python函数将输入的索引值变成输出的分组键
        '''
        display('df2','df2.groupby(str.lower).mean()')
        display('df','df.groupby(df["key"].str.lower()).mean()')
        '''
        任何前面的多个分组键可以组合并输出成一个多重索引的结果
        '''
        df2.groupby([mapping,str.lower]).mean()
        df2.groupby([str.lower,mapping]).mean()
        '''
        用于计算通过不同方法在不同年代发现的行星的个数
        '''
        decade = 10 * (planets['year'] // 10)
        decade = decade.astype(str) + 's'
        decade.name = 'decade'

        planets.groupby(['method',decade])['number'].sum().unstack().fillna(0)
        '''
        我们结合前面介绍过的多种操作之后，我们能在真实的数据集上完成多强大的操作。
        我们立即获得了过去几十年间我们是如何发现行星的大概统计
        '''
        '''
        要将Python中的字符串转换为小写，可以使用lower()方法
        将DataFrame中的某一列转换为小写，可以使用pandas库的str.lower()方法
            df['Name'] = df['Name'].str.lower()
        '''
        '''
        decade.name = 'decade'：这一行代码为新的变量（decade）
        设置一个名称，即'decade'。这样，在后续的数据处理过程中，
        可以通过这个名称来引用这个变量
        '''
        '''
        DataFrame.astype(dtype)
        DataFrame 是要进行数据类型转换的 Pandas 数据框对象，
        dtype 是要转换成的数据类型
        把 column_name 列的数据类型从默认的浮点数类型转换为整数类型
        astype() 函数可以应用于多个列，只需分别指定每个列的数据类型即可
        '''
        # 7.29
        '''
        在Python中，可以使用join()方法将一个字符串列表连接成一个字符串。
        join()方法的语法如下：
            str.join(iterable)
        str是分隔符，用于连接列表中的每个元素；iterable是一个可迭代对象，
        如列表、元组等
        '''
        # 7.30
        '''
        将数据透视表想象成一个*多维*版本的`GroupBy`聚合
        '''
        import pandas as pd
        import numpy as np
        import seaborn as sns

        titanic = pd.read_csv(r'C:\Users\陈泽鹏\Desktop\拆分合并\seaborn-data-master\titanic.csv')
        titanic.head()
        
        titanic.groupby('sex')[['survived']].mean()

        titanic.groupby(['sex','class'])['survived'].aggregate('mean').unstack()
        '''
        二维的`GroupBy`对于在Pandas中进行普通分组统计时是足够的，
        而透视表`pivot_table`，能简洁的处理这种多维度的聚合操作。
        '''
        titanic.pivot_table('survived',index='sex',columns='class')

        age = pd.cut(titanic['age'],[0,18,80])
        '''
        数值变量转为了名义变量
        '''
        titanic.pivot_table('survived',index=['sex',age],columns='class')
        # 7.31
        '''
        pd.cut() 是 pandas 库中的一个函数，用于将连续数值数据分割成离散的区间。
        pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, 
            include_lowest=False)
        x：需要分割的一维数组或Series。
        bins：用于分割的区间边界，可以是一个列表、数组或者一个整数。如果是整数，
            则表示等宽的区间。
        '''
        fare = pd.qcut(titanic['fare'], 2)
        titanic.pivot_table('survived',index=['sex',age],
                            columns=[fare,'class'])
        '''
        pd.qcut() 是 Pandas 库中的一个函数，用于将连续数值数据划分为相等数量
        的区间。它根据数据的分位数来划分区间，每个区间包含相同数量的数据点
        pd.qcut(x, q, labels=None, retbins=False)
        x: 要进行切分的一维数组或系列（Series）。
        q: 要划分的区间数量。可以是整数或一个包含分位数的列表。
        labels: 可选参数，用于指定每个区间的标签。如果提供，
            必须与区间数量相匹配。
        retbins: 可选参数，默认为 False。如果设置为 True，
            则返回每个区间的边界值。
        '''
        '''
        `DataFrame`的`pivot_table`方法的完整签名如下：
        pd.pivot_table
        (
            data, # DataFrame，当为方法时，这里是self
            values=None, # 用来聚合的列
            index=None, # 行索引，行分组的条件
            columns=None, # 列索引，列分组的条件
            aggfunc='mean', # 聚合函数，默认平均值
            fill_value=None, # NA值的替代值
            margins=False, # 总计，行与列相加的结果
            dropna=True, # 是否移除含有NA值的列
            margins_name='All', # 总计的行和列的标签
        )
        '''
        '''
        其中的`fill_value`和`dropna`与数据集的缺失值相关
        `aggfunc`参数指定数据透视表使用的聚合函数，默认是平均值`'mean'`。
        就像`GroupBy`中一样，聚合函数可以通过函数名称的字符串来指定
        （例如``'sum'``、``'mean'``、``'count'``、``'min'``、``'max'``等）。
        除此之外，也可以通过一个字典将列与聚合函数对应起来作为`aggfunc`的参数
        '''
        titanic.pivot_table(index='sex',columns='class',
                            aggfunc={'survived': 'sum','fare':'mean'})
        '''
        `values`参数也被忽略了；当我们将列和聚合函数映射的字典传递到`aggfunc`参数时，
        进行聚合的列显然是不需要指定的
        '''
        '''
        对每个组进行总计（或者小计）是很有用的。这可以通过指定`margins`参数来计算
        '''
        titanic.pivot_table('survived',index='sex',columns='class',margins=True)
        '''
        结果最后一行展示了所有性别不同舱位的存活率，
        最后一列展示了所有舱位不同性别的存活率，而右下角的数字代表总体存活率，
        约为38%。总计（或小计）的标签可以通过`margins_name`参数来制定，
        默认为`"All"`
        '''
        births = pd.read_csv(r'C:\Users\陈泽鹏\Desktop\拆分合并\Python_Data_Science_Handbook\notebooks\data\births.csv')
        births.head()

        births['decade'] = 10 * (births['year'] // 10)
        births.pivot_table('births',index='decade',columns='gender',
                           aggfunc='sum')
        '''
        我们会立刻发现男孩的出生人数在每一个年代都超过了女孩
        '''
        %matplotlib inline
        import matplotlib.pyplot as plt
        sns.set()
        births.pivot_table('births',index='year',columns='gender',
                           aggfunc='sum').plot()
        plt.ylabel('total births per year')
        Axis = plt.gca()
        Axis.yaxis.get_major_formatter().set_useOffset(False)
        Axis.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x))
            )
        '''
        plt.gac() 是matplotlib库中的一个函数，用于获取当前的axes对象。
        它通常在自定义图形绘制时使用，以便在当前轴上进行进一步的绘图操作
        '''
        '''
        '{:,.0f}'.format(x) 这段代码的意思是将变量 x 的值按照一定的格式进行输出。
        {:,.0f} 是一个格式说明符，它告诉Python如何格式化 x。
            {}：这是占位符，表示将被替换为 x 的值。
        :：这是一个可选的字段宽度指定符，如果提供了这个，
            那么输出的字符串将至少有这么多的字符宽。
        ,：这是一个可选的数字分组符号，用于在数字中添加逗号作为千位分隔符。
            例如，1000会被格式化为"1,000"。
        .0f：这是一个类型指定符，表示 x 应该是一个浮点数，
            并且小数部分应该被舍去（即，结果应为整数）
        '''
        '''
        应该对数据进行一定清洗，删除由于错误输入日期导致的离群值（例如6月31日）
        或者缺失值（例如6月99日）。一次性删除这些离群数据的简单办法是通过一种
        叫sigma-clipping的稳健统计操作
        '''
        quartiles = np.percentile(births['births'], [25, 50, 75])
        mu = quartiles[1]
        sig = 0.74 * (quartiles[2] - quartiles[0])
        '''
        np.percentile 是 NumPy 库中的一个函数，用于计算数组中给定百分位数的值
        numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, 
            interpolation='linear', keepdims=False)
        a：输入数组
        q：要计算的百分位数，范围在 0 到 100 之间
        axis：沿着哪个轴计算百分位数，默认为 None，表示计算整个数组的百分位数
        out：可选参数，用于存储结果的输出数组
        overwrite_input：可选参数，布尔值，表示是否允许修改输入数组
        interpolation：可选参数，插值方法，可选值为 'linear'（线性插值）、'lower'（下限）、'higher'（上限）和 'midpoint'（中点），默认为 'linear'
        keepdims：可选参数，布尔值，表示是否保持原数组的维度，默认为 False
        '''
        '''
        将年份乘以10000，月份乘以100，然后加上日期，将这三部分组合成一个整数。
        接着，使用pd.to_datetime()函数将这个整数转换为日期格式
        '''
        '''
        dayofweek 是一个用于获取日期中星期几的函数。
        '''
        # m = np.quantile(births['births'], [0.25, 0.50, 0.75])
        '''
        np.quantile() 是一个NumPy库中的函数，用于计算数组的分位数。
        # 计算第25个百分位数（即下四分位数）
        lower_quartile = np.quantile(data, 0.25)
        '''
        a = np.random.standard_normal(10000)
        iq = np.percentile(a, [25, 50, 75])
        num = 1 / (iq[2] - iq[0])
        '''
        四分位距（Interquartile Range，IQR），四分位数是将数据集分为四个等份的
        数值，其中第一四分位数（Q1）表示所有数值中最小的25%，第二四分位数（Q2）
        表示所有数值中50%的位置，第三四分位数（Q3）表示所有数值中最大的25%。
        计算 Q3 - Q1，我们可以得到 IQR 的值，即 1.34896。
        这个值可以用来衡量数据的离散程度，较大的 IQR 表示数据分布较为分散，
        较小的 IQR 表示数据分布较为集中。
        '''
        births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
        '''
        这里的@符号表示变量引用，mu和sig分别表示均值和标准差
        '''
        '''
        df.query() 是Pandas库中的一个函数，用于在DataFrame中进行条件查询。
        它允许你使用布尔表达式来筛选满足条件的行。
        df.query('Age > 28')
        '''
        # any(births['day'].isna())
        # l = [1,2,'a',None]
        # df = pd.DataFrame(l,columns=[1])
        # any(df.isna())
        '''
        any() 是 Python 中的一个内置函数，用于检查可迭代对象
        （如列表、元组等）中是否存在至少一个元素为真（True）。
        如果存在至少一个元素为真，则返回 True，否则返回 False
        '''
        births['day'] = births['day'].astype(int)

        births.index = pd.to_datetime(births['year']*10000 + 
                                      births['month']*100 +
                                      births['day'],format='%Y%m%d')
        
        births['dayofweek'] = births.index.dayofweek

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        births.pivot_table('births',index='dayofweek',columns='decade',
                           aggfunc='mean').plot()
        plt.gca().set_xlim(0, 6)
        plt.gca().set_xticklabels(['Mon','Tue','Wed','Thurs','Fri','Sat','Sun'])
        plt.ylabel('mean birth by day')
        '''
        使用plt.gca()获取当前的Axes对象，然后调用set_xticklabels()方法
        来设置x轴刻度标签
        '''
        '''
        # 设置X轴的范围
        plt.gca().set_xlim(min(x), max(x))
        '''
        '''
        出生数在休息日要比工作日少。还要注意到1990和2000年代数据缺失，
        原因是疾控中心的数据从1989年开始就只包含月份信息了
        '''
        '''
        分析每年每天的平均出生数
        '''
        births_by_date = births.pivot_table('births',index=[births.index.month,births.index.day])
        '''
        简单的绘制图表，我们可以将上面的月份日期随便放在一个闰年年份中形成
        完整的时间序列（闰年是为了保证2月29日也能包含在结果集中）
        '''
        births_by_date.index = [pd.datetime(2004,month,day) for (month, day) in 
                                births_by_date.index]
        '''
        pd.datetime() 是一个用于创建 Pandas 中的日期时间对象的函数。
        它接受一个或多个参数，用于指定年、月、日、时、分、秒等时间信息。
        import pandas as pd
        # 创建一个日期时间对象
        dt = pd.datetime(2023, 7, 3, 10, 30, 0)
        '''
        fig, ax = plt.subplots(figsize=(12, 4))
        births_by_date.plot(ax=ax)
        '''
        创建了一个图形（figure）和一个坐标轴（axes）对象。
        plt.subplots()函数用于创建一个图形和多个子图
        births_by_date.plot()函数用于绘制数据，
        ax=ax表示将数据绘制在指定的坐标轴上
        '''
        '''
        已经学习到的很多Python和Pandas的工具可以联合使用来深入分析不同的数据集
        以获得需要的结果
        '''
        '''
        list.pop() 是 Python 中用于从列表中移除并返回指定索引处的元素的方法。
        如果不提供索引，则默认移除并返回列表的最后一个元素。
        my_list = [1, 2, 3, 4, 5]
        last_element = my_list.pop()
        print(last_element)  # 输出：5
        print(my_list)  # 输出：[1, 2, 3, 4]
        '''
        '''
        要将字符串转换为列表，可以使用list()函数
        s = "hello"
        lst = list(s)
        print(lst)  # 输出：['h', 'e', 'l', 'l', 'o']
        '''
        '''
        要将列表转换为字符串，可以使用Python的join()方法。首先，确保列表中的所有元素都是
        字符串类型，然后使用join()方法将它们连接起来
        '''
        '''
        Given a range of numbered days, i...,jl and a number k. determine the number of days 
        in the range that arebeautiful. Beautiful numbers are defined as numbers where 
        i-reverse(i) is evenly divisible by k. lf a dav's value is abeautiful number, 
        it is a beautiful day., Return the number of beautiful days in the range.
        '''
        def beautifulDays(i, j, k):
            count = 0
            
            while i <= j:
                l = []
                m = str(i)
                m_l = list(m)

                for num in range(len(m_l)):
                    l.append(m_l.pop())
                
                if (i - int(''.join(l))) % k == 0:
                    count += 1
                i += 1
            
            return count
        
        # 8.5
        '''
        Python的一个强大的特点就是它能相对简单的处理和操作字符串数据。Pandas在此基础上提供了一整套
        *向量化字符串操作*，这成为了当我们处理（清洗）真实世界数据时非常关键的功能
        看它们在我们对从互联网采集到的非常不规范的数据集进行清洗时发挥的作用
        '''
        '''
        NumPy和Pandas的工具能向量化算术运算，可以很容易和快速的对数组的元素进行相同的数学计算
        '''
        import numpy as np

        x = np.array([2,34,56,7,8])
        x * 2
        '''
        这种*向量化*的操作能简化数组元素的操作语法：我们不再需要担心数组的大小和形状，
        只需要关注于需要进行的运算本身
        '''
        '''
        对于字符串数组，NumPy没有提供这种简单的操作，因此你需要继续使用循环语法来处理
        '''
        data = ['peter', 'Paul', 'MARY', 'gUIDO']
        [s.capitalize() for s in data]
        '''
        capitalize()是Python中的一个字符串方法，用于将字符串的第一个字符转换为大写，
        其余字符转换为小写
        '''
        # data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
        # [s.capitalize() for s in data]
        '''
        对于含有缺失值的数据集来说就出问题了
        '''
        '''
        Pandas包含了前面说到的向量化的字符串操作，而且还能正确的处理缺失值，
        这可以通过Pandas的Series和Index对象的`str`属性来实现
        '''
        import pandas as pd
        name = pd.Series(data)
        name.str.capitalize()
        '''
        在IPython中在`str`属性上使用制表符自动补全功能可以列出Pandas中支持的
        所有的向量化字符串操作
        '''
        monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
        '''
        几乎所有Python內建的字符串方法都有Pandas的向量化版本。
        下面是Pandas的`str`属性中与Python內建字符串方法一致的方法：
        |             |                  |                  |                  |
        |-------------|------------------|------------------|------------------|
        |``len()``    | ``lower()``      | ``translate()``  | ``islower()``    | 
        |``ljust()``  | ``upper()``      | ``startswith()`` | ``isupper()``    | 
        |``rjust()``  | ``find()``       | ``endswith()``   | ``isnumeric()``  | 
        |``center()`` | ``rfind()``      | ``isalnum()``    | ``isdecimal()``  | 
        |``zfill()``  | ``index()``      | ``isalpha()``    | ``split()``      | 
        |``strip()``  | ``rindex()``     | ``isdigit()``    | ``rsplit()``     | 
        |``rstrip()`` | ``capitalize()`` | ``isspace()``    | ``partition()``  | 
        |``lstrip()`` |  ``swapcase()``  |  ``istitle()``   | ``rpartition()`` |
        '''
        '''
        要提醒的是，这些方法与內建字符串方法可能有着不同的返回值，
        如`lower()`返回的是一个字符串的Series对象
        '''
        monte.str.lower()
        '''
        另外一些返回的是数字的Series对象
        '''
        monte.str.len()
        '''
        或布尔值的Series对象
        '''
        '''
        monte.str表示对名为"monte"的Pandas DataFrame或Series中的所有字符串进行操作
        '''
        monte.str.startswith('T')
        '''
        还有一些会返回诸如列表那样的复合类型的Series对象
        '''
        monte.str.split()
        '''
        后面会讨论到如何操作这种列表组成的Series对象
        '''
        '''
        一些方法可以接受正则表达式来检查每个元素字符串是否匹配模式，它们遵从Python內建的`re`模块的API规范：
        | 方法 | 描述 |
        |--------|-------------|
        | ``match()`` | 在每个元素上调用``re.match()``方法，返回布尔类型Series |
        | ``extract()`` | 在每个元素上调用``re.match()``方法，返回匹配到模式的正则分组的Series |
        | ``findall()`` | 在每个元素上调用``re.findall()``方法 |
        | ``replace()`` | 将匹配模式的字符串部分替换成其他字符串值 |
        | ``contains()`` | 在每个元素上调用``re.search()``，返回布尔类型Series |
        | ``count()`` | 计算匹配到模式的次数 |
        | ``split()``   | 等同于``str.split()``，但是能接受正则表达式参数 |
        | ``rsplit()`` | 等同于``str.rsplit()``, 但是能接受正则表达式参数 |
        '''
        monte.str.extract('([A-Za-z]+)',expand=False)
        '''
        str.extract()是一个字符串方法，用于从Series中的每个元素（即字符串）中
        提取匹配正则表达式的子串
        正则表达式([A-Za-z]+)表示匹配一个或多个连续的字母（不区分大小写）
        expand=False表示返回的结果将是一个DataFrame，而不是一个Series。
        如果设置为True，则返回的结果将是一个Series，其中每个元素都是一个列表，
        包含所有匹配的子串
        '''
        monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
        '''
        str.findall()方法，该方法会返回一个包含所有匹配结果的列表
        '''
        '''
        ^表示字符串的开头
        [^AEIOU]表示不是大写元音字母A、E、I、O、U的任何字符
        .*表示任意数量的任意字符（包括0个）
        [^aeiou]表示不是小写元音字母a、e、i、o、u的任何字符
        $表示字符串的结尾
        '''
        '''
        在`Series`或`DataFrame`上简洁的应用正则表达式的特性，
        在清洗和分析数据任务中非常有用
        '''
        '''
        下面是一些无法分类的其他方法但也是很方便的字符串功能：
        | 方法 | 描述 |
        |--------|-------------|
        | ``get()`` | 对每个元素使用索引值获取字符中的字符 |
        | ``slice()`` | 对每个元素进行字符串切片 |
        | ``slice_replace()`` | 将每个元素的字符串切片替换成另一个字符串值 |
        | ``cat()``      | 将所有字符串元素连接成一个字符串 |
        | ``repeat()`` | 对每个字符串元素进行重复操作 |
        | ``normalize()`` | 返回字符串的unicode标准化结果 |
        | ``pad()`` | 字符串对齐 |
        | ``wrap()`` | 字符串换行 |
        | ``join()`` | 字符串中字符的连接 |
        | ``get_dummies()`` | 将字符串按照分隔符分割后形成一个二维的dummy DataFrame |
        '''
        '''
        `get()`和`slice()`操作，可以对每个字符串元素进行索引访问和切片的操作
        '''
        monte.str.slice(0,3)
        '''
        可以通过`str.slice(0, 3)`获取每个字符串元素的前三个字母。
        也可以通过Python标准的切片语法来完成，`df.str[:3]`等同于
        `df.str.slice(0, 3)`
        '''
        monte.str[:3]
        '''
        索引取值操作也是一样，`df.str[i]`等同于`df.str.get(i)`
        '''
        '''
        `get()`和`slice()`方法还能支持对`split()`返回的列表进行取值操作
        '''
        monte.str.split().str.get(-1)
        '''
        `get_dummies()`。这个方法在你的数据中含有某种编码的指示器的时候非常有用
        '''
        full_monte = pd.DataFrame({'name': monte,
                                   'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})   
        '''
        `get_dummies()`方法能让你快速的将这些编码的指示器变量分解出来，
        并形成一个`DataFrame`
        '''
        '''
        .get_dummies('|')：这是在对选定的列进行one-hot编码。
        get_dummies函数会将每个唯一的字符串值转换为一个新的二进制列（0或1），
        如果原始字符串包含该值，则新列的值为1，否则为0。
        这里的'|'是一个分隔符，表示原始字符串中的每个部分都应该被视为一个独立的值。
        例如，如果原始字符串是"a|b|c"，那么生成的新列将是"a", "b", "c"
        '''
        final_monte = full_monte['info'].str.get_dummies('|')

        for col in full_monte.columns:
            final_monte[col] = full_monte[col]

        new_order = ['name','info','A','B','C','D']

        final_monte = final_monte[new_order]
        '''
        有了上述的这些向量化字符串方法，你可以在清洗数据时构建无穷无尽的字符串处理
        流程
        '''
        '''
        这些向量化字符串操作是我们对不规范的真实世界数据进行清洗的最有效工具
        '''
        '''
        目标是将这些菜谱数据解析成配方的列表，这样我们就能很快速的根据我们手头的材料找到相应配方
        的菜谱
        '''
        # !curl -O https://s3.amazonaws.com/openrecipes/20170107-061401-recipeitems.json.gz
        # !gunzip 20170107-061401-recipeitems.json.gz

        try:
            recipes = pd.read_json(r'C:\Users\陈泽鹏\Desktop\拆分合并\Python_Data_Science_Handbook\notebooks\data\20170107-061401-recipeitems.json\20170107-061401-recipeitems.json')
        except ValueError as e:
            print('ValueError:', e)
        '''
        这里会产生一个`ValueError`指出有冗余的数据。原因是这个文件*每一行*都是一个正确的JSON，
        但是整个文件不是正确的JSON格式。我们来验证一下
        '''
        '''
        在Python中，当一个字符串前面加上字母r时，它就会被解释为一个原始字符串。
        这意味着，在这个字符串中的反斜杠（\）不会被当作转义字符来处理，
        而是被直接视为普通字符。这在某些情况下非常有用，比如当你需要在字符串中包含一些
        特殊字符（如反斜杠、引号等），但又不希望这些字符被Python解释器解析时，
        就可以使用原始字符串
        '''
        # import bz2

        with open(r'C:\Users\陈泽鹏\Desktop\拆分合并\Python_Data_Science_Handbook\notebooks\data\20170107-061401-recipeitems.json\20170107-061401-recipeitems.json') as f:
            line = f.readline()
        pd.read_json(line).shape
        '''
        在Python中读取bz2文件，可以使用bz2库：

        import bz2
        # 用二进制模式打开文件
        with bz2.open('file.bz2', 'rt') as f:
            content = f.read()
            print(content)

        这段代码首先导入bz2库，然后使用bz2.open()函数以文本模式（'rt'）
        打开bz2文件
        '''
        '''
        通过读取文件一行我们验证了我们的想法，现在我们需要将这些正确的JSON行
        合并在一起。实现这个目标的一种方式就是我们手动将所有的行合并成一个
        JSON Array，然后将这个JSON Array的字符串传递到`pd.read_json`来进行解析
        '''
        # import codecs
        # 8.6
        with open(r'C:\Users\陈泽鹏\Desktop\拆分合并\Python_Data_Science_Handbook\notebooks\data\20170107-061401-recipeitems.json\20170107-061401-recipeitems.json',
                  'r', encoding='utf-8') as f:
            data = (line.strip() for line in f)
            json_data = "[{0}]".format(','.join(data))            
        
        recipes = pd.read_json(json_data)
        '''
        data = (line.strip() for line in f)：将处理后的每行数据组成一个生成器对象，
        赋值给变量data
        生成器对象是一种特殊类型的迭代器，它允许你在需要时生成值，而不是一次性生成所有值。
        这在处理大量数据时非常有用，因为它可以减少内存消耗
        '''
        recipes.shape
        recipes.iloc[0]
        '''
        结果中有很多的数据，但大多数列的数据都是混乱不堪的，正如所有从网络中爬取的数据
        一样。特别注意到，配方列表是一个字符串的格式，因此我们需要特别小心的在我们
        感兴趣的列中进行数据提取操作
        '''
        '''
        dtype('O')表示对象类型，通常用于表示Python中的列表、元组、字典等数据结构。
        在NumPy库中，它用于表示任意类型的数据，包括字符串、整数、浮点数等。
        '''
        recipes.ingredients.str.len().describe()
        '''
        配方的列表平均有250个字符长，最短的是0个字符，而最长的能达到接近10000个字符
        '''
        '''
        idxmax()是Pandas库中的一个函数，用于返回DataFrame或Series对象中最大值的索引。
        如果传入参数，那么它将返回指定轴上最大值的索引。
        '''
        recipes['name'][recipes.ingredients.str.len().idxmax()]

        recipes['description'].str.contains('[Bb]reakfast').sum()

        recipes.ingredients.str.contains('[Cc]innamon').sum()

        recipes.ingredients.str.contains('[Cc]inamon').sum()
        '''
        这些类型的基础数据分析工作，都可以通过Pandas的字符串工具进行并获得结果。
        这正是Python在数据科学领域优于其他语言的地方
        '''
        '''
        给定一系列的原材料组成的配方，找到应用了所有原料的菜谱。虽然看起来很容易，
        实际上这个任务的难点在于数据的异构性：即无法找到一个简单的操作，
        能从每一行中提取出干净的原料列表
        '''
        spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
        
        import re
        '''
        dict() 是 Python 中的一个内置函数，用于创建字典。创建一个空字典的方法如下：
        my_dict = dict()
        初始化一些键值对:
        my_dict = dict(key1='value1', key2='value2', key3='value3')
        '''
        spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE)) for spice in spice_list))
        spice_df.head()

        selection = spice_df.query('parsley & paprika & tarragon')
        len(selection)

        recipes.name[selection.index]
        '''
        使用Pandas的字符串方法我们可以对数据进行异常方便的清洗操作。当然如果希望构建一个成熟的
        菜谱推荐系统的话，需要比上例*复杂的多*的技巧和工程。将每个菜谱中的原料配方提取出来变成
        一个列表会是其中很重要的一环；不幸的是，因为数据格式的多样性，这项任务会相对很耗时。
        清洗和预处理真实世界的数据是这个领域非常主要的工作之一，Pandas提供了一些工具能帮助你
        很有效率的完成它
        '''
        '''
        Complete the saveThePrisoner function in the editor below. 
        It should return an integer representing the chair number of the prisoner to warn.
        saveThePrisoner has the following parameter(s):
        int n: the number of prisoners
        int m: the number of sweets
        int s: the chair number to begin passing out sweets from
        Returns
        int: the chair number of the prisoner to warn
        '''
        def saveThePrisoner(n, m, s):
            last_sweet = (m + s - 1) % n 

            if last_sweet == 0:
                return n
            else:
                return last_sweet



        






        





        
    




        




        

        


        





        


    










        
        

        
        


        



    except Exception as e:
        print(e)
