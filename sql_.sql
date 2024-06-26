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




