'''
创建数据用表
'''
drop table if exists tmp.xxx;
create table tmp.xxx(
	id serial primary key,
	user_name varchar(50) not null,
	birthday date,
	is_active boolean default true
);
insert into tmp.xxx(user_name,birthday,is_active)values('ba','1999-07-10',False),('edf','2000-12-21',true),('z','1997-12-09',False),('m','2006-10-29',true)
	,('a','1999-07-10',False);
	

'''
Find the difference between the total number of CITY entries in the table 
and the number of distinct CITY entries in the table.
'''
select (count(1)-count(distinct user_name)) as cnt from xxx

'''
char_length(str)
计算单位：字符
不管汉字还是数字或者是字母都算是一个字符
length(str)
计算单位：字节
utf8编码：一个汉字三个字节，一个数字或字母一个字节。
gbk编码：一个汉字两个字节，一个数字或字母一个字节。
'''
select city,length(city) as sm_ from station order by length(city), city asc  limit 1;
select city,length(city) as lar_ from station order by length(city) desc, city asc limit 1;

'''
Query the list of city names starting from vowels(i.e.a,e,i,o or u)from station
left(str,length)\right(str,length)\substring(str, pos, length)\substring_index(被截取字符串，关键字，关键字出现的次数)
'''
select distinct city from station where left(city,1) in ('a','e','i','o','u');

'''
Write a query identifying the type of each record in the TRIANGLES table using its three side lengths. Output one of the following statements for each record in the table:
Equilateral: It's a triangle with  sides of equal length.
Isosceles: It's a triangle with  sides of equal length.
Scalene: It's a triangle with  sides of differing lengths.
Not A Triangle: The given values of A, B, and C don't form a triangle.
'''
select case when (A = B and B = C) then 'Equilateral' 
    when (A = B or B = C or A = C) and (A + C > B and B + C > A and A + B > C) then 'Isosceles'
    when (A + B > C and A + C > B and B + C > A) then 'Scalene'
    else 'Not A Triangle' end
from triangles

'''
Query an alphabetically ordered list of all names in OCCUPATIONS, immediately followed by the first letter of each profession as a parenthetical 
(i.e.: enclosed in parentheses). For example: AnActorName(A), ADoctorName(D), AProfessorName(P), and ASingerName(S).
Query the number of ocurrences of each occupation in OCCUPATIONS. Sort the occurrences in ascending order, and output them in the following format:
There are a total of [occupation_count] [occupation]s.
where [occupation_count] is the number of occurrences of an occupation in OCCUPATIONS and [occupation] is the lowercase occupation name. 
If more than one Occupation has the same [occupation_count], they should be ordered alphabetically.
'''
select concat(Name,"","(",left(Occupation,1),")") from occupations order by Name asc;
select concat("There are a total of"," ",cast(count(Name) as char)," ",concat(lower(Occupation),"s.")) 
from occupations group by Occupation order by count(Occupation) asc, Occupation asc;

'''
SQL语句 with as 用法
WITH AS短语，也叫做子查询部分（subquery factoring），是用来定义一个SQL片断，该SQL片断会被整个SQL语句所用到。这个语句算是公用表表达式（CTE）。
比如 with A as (select * from class)
		select *from A  
这个语句的意思就是，先执行select * from class   得到一个结果，将这个结果记录为A  ，在执行select *from A  语句。A 表只是一个别名。
也就是将重复用到的大批量 的SQL语句，放到with  as 中，加一个别名，在后面用到的时候就可以直接用。
对于大批量的SQL数据，起到优化的作用。
'''
'''
Pivot the Occupation column in OCCUPATIONS so that each Name is sorted alphabetically and displayed underneath its corresponding Occupation. 
The output column headers should be Doctor, Professor, Singer, and Actor, respectively.
Note: Print NULL when there are no more names corresponding to an occupation.
Jenny    Ashley     Meera  Jane
Samantha Christeen  Priya  Julia
NULL     Ketty      NULL   Maria
'''
with tmp as(
    select *,row_number() over (partition by Occupation order by name) as row_num
    from occupations
)
select max(case when Occupation='Doctor' then name end) as Doctor
    ,max(case when Occupation='Professor' then name end) as Professor
    ,max(case when Occupation='Singer' then name end) as Singer
    ,max(case when Occupation='Actor' then name end) as Actor
from tmp 
group by row_num

'''
You are given a table, BST, containing two columns: N and P, where N represents the value of a node in Binary Tree, and P is the parent of N.
Write a query to find the node type of Binary Tree ordered by the value of the node. Output one of the following for each node:
Root: If node is root node.
Leaf: If node is leaf node.
Inner: If node is neither root nor leaf node.
'''
select case 
    when P is NUll then concat(cast(N as char),' ','Root')
    when N not in (select distinct P from BST where p is not NULL) then concat(cast(N as char),' ','Leaf')
    else concat(cast(N as char),' ','Inner') 
    end
from BST
order by N
'''
Write a query calculating the amount of error (i.e.:  average monthly salaries), and round it up to the next integer
'''
'''
CAST函数语法规则是：Cast(字段名 as 转换的类型 )，其中类型可以为：
	CHAR[(N)] 字符型
	DATE 日期型
	DATETIME 日期和时间型
	DECIMAL float型
	SIGNED int
	TIME 时间型
	
	整数 : SIGNED
	无符号整数 : UNSIGNED（非负整数）
'''
'''
REPLACE(str, from_str, to_str)
	str:要进行替换操作的原始字符串。
	from_str:需要被替换的子串。
	to_str:用于替换from_str的新子串。
'''
select ceil(avg(salary) - avg(cast((replace(cast(salary as char),'0','')) as signed)))
from employees

select ceil(avg(salary) - avg(replace(salary,'0','')))
from employees

'''
Write a query to find the maximum total earnings for all employees as well as the total number of employees who have maximum total earnings
'''
select max(months*salary), count(*)
from Employee
group by months*salary
order by months*salary desc
limit 1

《==》

select months*salary, count(*)
from Employee
group by months*salary
order by months*salary desc
limit 1

'''
Query the smallest Northern Latitude (LAT_N) from STATlON that is greater than 38.7780. Round your answer to 4 decimal places.
'''
select round(lat_n,4) from station where lat_n > 38.7780 order by lat_n limit 1

'''
曼哈顿距离的正式意义为L1-距离或城市区块距离，也就是在欧几里德空间的固定直角坐标系上两点所形成的线段对轴产生的投影的距离总和
在平面上，坐标(x1,y1)的i点与坐标(x2,y2)的j点的曼哈顿距离为：d(i,j)=|X1-X2|+|Y1-Y2|
'''

'''
Consider Pi (a, c) and Pi (b, d) to be two points on a 2D plane where (a, b) are the respective minimum and maximumvalues of Northern Latitude (LAT_N) 
and (c, d) are the respective minimum and maximum values of Western Longitude(LONG_W) in STATION.
Query the Euclidean Distance between points P and P, 
'''
select round(sqrt(pow(min(lat_n)-max(lat_n),2) + pow(min(long_w)-max(long_w),2)),4) from station
'''
在MySQL中，可以使用POW()函数进行平方运算，使用SQRT()函数进行平方根运算。pow() 是 MySQL 中的一个函数，用于计算一个数的指数次幂，
POW(base, exponent)，其中，base 是底数，exponent 是指数。	
'''

'''
在PostgreSQL中，JSONB是一种二进制格式的JSON数据类型，它允许你在数据库中存储和查询复杂的JSON数据结构。
与普通的JSON类型相比，JSONB在存储时会将JSON数据解析为二进制格式，这使得查询性能更优，并支持索引。
'''
select *, (return_data -> 'data' -> 'ENT_INFO' ->> 'EXCEPTIONLIST') as EXCEPTIONLIST
from zebra__gateway_log__i_d 
where application_name='http调用:fpdxw' 
and short_name='bidata' 
and (return_data -> 'data' -> 'ENT_INFO' ->> 'EXCEPTIONLIST')::jsonb != '[]'
order by create_time desc limit 100

select *, (return_data -> 'data' -> 'ENT_INFO' ->> 'EXCEPTIONLIST') as EXCEPTIONLIST
from zebra__gateway_log__i_d 
where application_name='http调用:fpdxw' 
and short_name='bidata' 
and (return_data -> 'data' -> 'ENT_INFO' ->> 'EXCEPTIONLIST') != '[]'
order by create_time desc limit 100

'''
A median is defined as a number separating the higher half of a data set from the lower half. 
Query the median of the Northern Latitudes (LAT_N) from STATION and round your answer to 4 decimal places.
'''
'''
窗口函数可以进行排序、生成序列号等一般的聚合函数无法实现的高级操作；聚合函数将结果集进行计算并且通常返回一行。窗口函数也是基于结果集的运算。
与聚合函数不同的是，窗口函数并不会将结果集进行分组合并输出一行；而是将计算的结果合并到基于结果集运算的列上。
'''
'''
**可作为窗口函数的函数分类: **
聚合函数：① 聚合函数（SUM、AVG、COUNT、MAX、MIN）
内置函数：② RANK、DENSE_RANK、ROW_NUMBER 等专用窗口函
'''
'''
***
	OVER 子句中的 ORDER BY只是用来决定窗口函数按照什么样的顺序进行计算的，对结果的排列顺序并没有影响；
	而需要在select的最后指定排序，不然整个结果集不确定顺序。主意：这两个order by的作用和意思完全不同。
***
'''
'''
我们得到的并不仅仅是合计值，而是按照ORDER BY子句指定的product_id的升序进行排列，计算出商品编号“小于自己”的商品的销售单价的合计值。
因此，计算该合计值的逻辑就像金字塔堆积那样，一行一行逐渐添加计算对象。
在按照时间序列的顺序，计算各个时间的销售额总额等的时候，通常都会使用这种称为累计的统计方法。
'''
select rc1_cnt,rk,sum(rc1_cnt) over (order by rk) as cnt
from
(
	select rc1_cnt, row_number() over(order by rc1_cnt) as rk
	from finorder_microcredit_rcrate_a_d
) as t
where (cnt % 2 = 0 and rk = cnt / 2)
or (cnt % 2 = 1 and rk = (cnt + 1) / 2)

select round(lat_n,4)
from
(
    select lat_n, row_number() over(order by lat_n) as rk, count(1) over () as cnt
    from station
) as t
where (cnt % 2 = 0 and rk = cnt / 2 end

'''
Given the ClTY and COUNTRY tables, query the names of all the continents (COUNTRY.Continent) and their respective
average city populations (ClTY,Population) rounded down to the nearest integer.
'''
select t2.continent, floor(avg(t1.population)) from city as t1 left join country as t2 on t1.countrycode=t2.code where t2.continent is not NULL group by t2.continent 
'''
MySQL中的FLOOR()函数用于向下取整，即返回小于或等于给定参数的最大整数
'''
SELECT FLOOR(10.5);
'''
检查一个字段是否不为空，可以使用IS NOT NULL条件
'''
SELECT * FROM users WHERE email IS NOT NULL;
