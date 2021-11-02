---
layout: post
title: MySQL Tutorial for Beginners
date: 2021-08-26 13:04
comments: true
external-url:
categories: Computer
---

## 安装

**brew安装Mysql**

```no
lihan@LideMacBook-Air ~ % brew install mysql
==> Installing mysql
==> Summary
🍺  /opt/homebrew/Cellar/mysql/8.0.26: 303 files, 296.7MB
==> mysql
We've installed your MySQL database without a root password. To secure it run:
    mysql_secure_installation

MySQL is configured to only allow connections from localhost by default

To connect run:
    mysql -uroot

To have launchd start mysql now and restart at login:
  brew services start mysql
Or, if you don't want/need a background service you can just run:
  mysql.server start
lihan@LideMacBook-Air ~ % mysql.server start
Starting MySQL
.. SUCCESS! 
lihan@LideMacBook-Air ~ % mysql_secure_installation

Securing the MySQL server deployment.

Connecting to MySQL using a blank password.

VALIDATE PASSWORD COMPONENT can be used to test passwords
and improve security. It checks the strength of password
and allows the users to set only those passwords which are
secure enough. Would you like to setup VALIDATE PASSWORD component?

Press y|Y for Yes, any other key for No: y

There are three levels of password validation policy:

LOW    Length >= 8
MEDIUM Length >= 8, numeric, mixed case, and special characters
STRONG Length >= 8, numeric, mixed case, special characters and dictionary                  file

Please enter 0 = LOW, 1 = MEDIUM and 2 = STRONG: 2
Please set the password for root here.

New password: 

Re-enter new password: 

Estimated strength of the password: 100 
Do you wish to continue with the password provided?(Press y|Y for Yes, any other key for No) : y
By default, a MySQL installation has an anonymous user,
allowing anyone to log into MySQL without having to have
a user account created for them. This is intended only for
testing, and to make the installation go a bit smoother.
You should remove them before moving into a production
environment.

Remove anonymous users? (Press y|Y for Yes, any other key for No) : y
Success.


Normally, root should only be allowed to connect from
'localhost'. This ensures that someone cannot guess at
the root password from the network.

Disallow root login remotely? (Press y|Y for Yes, any other key for No)  n    
 ... skipping.
By default, MySQL comes with a database named 'test' that
anyone can access. This is also intended only for testing,
and should be removed before moving into a production
environment.


Remove test database and access to it? (Press y|Y for Yes, any other key for No) : y
 - Dropping test database...
Success.

 - Removing privileges on test database...
Success.

Reloading the privilege tables will ensure that all changes
made so far will take effect immediately.

Reload privilege tables now? (Press y|Y for Yes, any other key for No) : y
Success.

All done! 
lihan@LideMacBook-Air ~ % mysql -u root -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 11
Server version: 8.0.26 Homebrew

Copyright (c) 2000, 2021, Oracle and/or its affiliates.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> exit;
Bye
```

**建立数据库**

```no
链接：https://pan.baidu.com/s/1xDsDWFLiRITYTT6HoFb7Tw 
提取码：s5iz
```

注意存放位置如果有中文路径可能会在之后报`error 2`错误

在`source`后直接拖入`sql`文件即可

```no
mysql> create database mydatabase;
Query OK, 1 row affected (0.01 sec)

mysql> use mydatabase
Database changed
mysql> source /Users/lihan/Desktop/summer/create-databases.sql 
Query OK, 0 rows affected, 1 warning (0.00 sec)
....
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mydatabase         |
| mysql              |
| performance_schema |
| sql_hr             |
| sql_inventory      |
| sql_invoicing      |
| sql_store          |
| sys                |
+--------------------+
9 rows in set (0.00 sec)

mysql> drop database mydatabase;
Query OK, 0 rows affected (0.01 sec)
```

## 单表检索

- Mysql大小写不分，建议大写关键字，小写其他内容
-  在行前使用`--`表示注释
- 每个语句后使用`;`表示结束
- 可以使用算术表达式
- 可以使用`AS`关键字为列取别名

```sql
SELECT first_name, point * 10 + 100 AS `discount_factor`
FROM customers
-- WHERE custom_id = 1
ORDER BY first_name;
```

- ` DISTINCT`可以删除重复项， *重复*是以列的组合判断的，`DISTINCT`必须放在开头

```sql
SELECT DISTINCT state 
FROM customers;
```

- `WHERE`中对于文本需要使用单引号或双引号，文本大小写不敏感
- 注意此处的`Customers`与`va`均大小写不敏感
- `!=` ，`<>`均意为不等于
- 日期类型数据也需使用引号
- 可以使用`AND`，`OR`，`NOT`创建复合条件语句
- 条件语句中可以使用算数表达式

```sql
SELECT *
FROM Customers
WHERE state <> 'va' 
			AND birth_data > '1990-01-01';
```

- `IN`关键字判断是否在集合中
- `BETWEEN`关键字判断是否在闭区间内

```sql
SELECT *
FROM customers
WHERE state IN ('VA', 'FL') 
			AND points BETWEEN 1000 AND 2000;
```

- `LIKE`关键字用于文本模糊匹配常用的有：`_`单一字符，`%`任意数目字符

```sql
SELECT *
FROM customers
WHERE last_name LIKE '%field%';
```

- `REGEXP`关键字使用正则表达式进行模糊匹配

```sql
SELECT *
FROM customers
WHERE last_name REGEXP 'field';
```

- `NULL`表示空值

```sql
SELECT *
FROM customers
WHERE phone IS NOT NULL;
```

- 结果集默认以主键为标准进行生序排序，使用`ORDER BY`关键字可以使用其他列为标准排序
- 默认升序排序，使用`DESC`进行降序排序
- 可以使用多个排序规则，越靠前优先级越高
- Mysql中允许用于排序的列不是结果集的列

```sql
SELECT first_name
FROM customers
ORDER BY state DESC, first_name DESC;
```

- `LIMIT`关键字用于限制结果集大小，可用于网站分页
- `LIMIT 3`代表结果集大小为3，`LIMIT 6, 3`代表跳过6行取3行
- 注意`WHERE`，`ORDER BY`，`LIMIT`在语句中出现的先后顺序

```sql
SELECT *
FROM customers
WHERE state <> 'va' 
ORDER BY points DESC
LIMIT 3;
```

## 多表检索

- `JOIN`默认代表内连接即 `INNER JOIN`
- 可以为表取别名

```sql
SELECT o.customer_id
FROM orders o
JOIN customers
		 ON o.customer_id = customer.customer_id;
```

- 可以对不同数据库的表进行连接

```sql
SELECT *
FROM order_items oi
JOIN sql_inventory.products p
		 ON oi.product_id = p.product_id;
```

- 表也可以与自身进行连接，称为自连接

```sql
SELECT e.employee_id,
			 e.first_name,
			 m.first_nam
FROM employees e
JOIN employees m
		 WHERE e.reports_to = m.employee_id;
```

- `JOIN`多次使用可以多表连接


```sql
SELECT *
FROM orders o
JOIN customers c
ON o.customer_id = c.customer_id
JOIN order_statuses os
ON o.status = os.order_status_id;
```

-  `AND`创建复合连接条件

```sql
SELECT *
FROM order_items oi
JOIN order_items_notes oin
		ON oi.order_id = oin.order_id
		AND oi.product_id = oin.product_id;
```

- 一般我们使用显式连接（第一句）而不是隐式连接（第二句）
- 显式连接不加`ON`指名连接条件或隐式连接不加`WHERE`，的时候称为交叉连接，返回笛卡尔积的结果

```sql
SELECT *
FROM orders o
JOIN customers c
		ON o.customer_id = c.customer_id;
		
SELECT *
FROM orders o, customers c
WHERE o.customer_id = c.customer_id;

SELECT *
FROM orders o, customers c;
```

- `LEFT JOIN`左外连接表示所有左表的记录都会返回，如果右表没有相匹配的结果，对应位置置为NULL
- `OUTER`关键字一般省去
- 为使得语句清晰，外连接尽量使用左连接

```sql
SELECT *
FROM customers c
LEFT OUTER JOIN orders o
		ON c.customer_id = o.customer_id;
```

- 多表连接时候，判断左右表关系似乎是按照`ON`来的，有待补充

```sql
SELECT *
FROM customers c
LEFT JOIN orders o
		ON c.customer_id = o.customer_id
LEFT JOIN shippers sh
		ON o.shipper_id = sh.shipper_id
ORDER BY c.customer_id;
```

- 和内连接类似，外连接也可以自连接

```sql
SELECT *
FROM employees m
LEFT JOIN emplyees m
		ON e.reports_to = m.emplyee_id;
```

- 当两个表的列名完全一样时，使用`USING`简化，`USING`同样适用于有多个列作为键码的情况

```sql
SELECT *
FROM orders o
JOIN customers c
		-- ON o.customer_id = c.customer_id
		USING (customer_id);

SELECT *
FROM order_items oi
JOIN order_items_notes oin
		-- ON oi.order_id = oin.order_id
		-- AND oi.product_id = oin.product_id;
		USING (order_id, product_id);
```

- `NATURAL JOIN`自然连接，使用同名的属性组，并且在结果中把重复的属性列去掉，不建议使用

```sql
SELECT *
FROM orders o
NATURAL JOIN customers c;
```

- `CROSS JOIN`交叉连接，返回笛卡尔积，即所有的组合结果
- 第一句为显式写法，第二句为隐式写法

```sql
SELECT *
FROM orders o
CROSS JOIN customers c;

SELECT *
FROM orders o, customers c;
```

- `UNION`联合查询结果，两次`SELECT`必须有相同的属性数，属性名以第一次查询为主进行整合

```sql
SELECT first_name
FROM archived_orders
UNION
SELECT name
FROM orders;
```

## 插入、更新、删除

- `INSERT INTO`插入单行
- `DEFAULT`可用于存在默认值或递增的属性
- `NULL`可用于允许空值的属性
- 可以指定要插入的列

```sql
INSERT INTO customers
VALUES (
  	DEFAULT, 
  	'John', 
  	'Smith', 
  	NULL,
		'address',
		'city',
		'CA',
		200);

INSERT INTO customers (
		first_name,
		last_name,
		birth_date,
		address,
		city,
		state)
VALUES (
  	'John', 
  	'Smith', 
  	'1990-01-01',
		'address',
		'city',
		'CA');
```

- 使用`VALUES`也可以一次插入多行
- 在删除一些行的情况下，自增的属性值依然会记住他们，即出现递增属性不连续

```sql
INSERT INTO shippers (name)
VALUES ('Shipper1'),
			 ('Shipper2'),
			 ('Shipper3');
```

- `LAST_INSERT_TO()`获得最新执行成功的`INSERT`语句的自增id

```sql
INSERT INTO orders (customer_id, order_date, status)
VALUES (1, '2019-01-01', 1);

INSERT INTO order_items
VALUES 
    (last_insert_id(), 1, 2, 2.5),
    (last_insert_id(), 2, 5, 1.5);
```

- `CREATE TABLE`可以通过`SELECT`语句创建基于查询的复制表，属性不再拥有自增，主键等性质

```sql
CREATE TABLE orders_archived AS
SELECT * FROM orders;
```

- `UPDATE`关键字可用于更新表中的某些行

```sql
UPDATE invoices
SET payment_total = 0.5 * invoice_total, 
    payment_date = '2019-01-01'
WHERE invoice_id = 3;
```

- `UPDATE`操作也可以结合子查询使用
- 子查询添加括号确保先执行
- 子查询返回多个数据时应使用`IN`关键字

```sql
UPDATE invoices
SET payment_total = 567, 
		payment_date = due_date
WHERE client_id = 
            (SELECT client_id 
            FROM clients
            WHERE name = 'Yadel');

UPDATE invoices
SET payment_total = 567, 
		payment_date = due_date
WHERE client_id IN 
            (SELECT client_id 
            FROM clients
            WHERE state IN ('CA', 'NY'));
```

- `DELETE`关键字用于删除
- `WHERE`可选，省略将删除表的所有记录

```sql
DELETE FROM invoices
WHERE client_id = 
            (SELECT client_id  
            FROM clients
            WHERE name = 'Myworks');
```

## 汇总数据

- Mysql内置有`MAX()`，`MIN()`，`AVG()`，`SUM()`，`COUNT()`等聚合函数
- 聚合函数不仅可以用于列，也可以用于表达式
- 聚合函数会忽略空值，对于`COUNT()`函数需要注意
- `COUNT(*) `不会忽略空值，`*`表示全部数据
- 可以使用`DISTINCT`筛掉列的重复值

```sql
SELECT 
    MAX(invoice_date) AS latest_date,  
    MIN(invoice_total) lowest,
    AVG(invoice_total) average,
    SUM(invoice_total * 1.1) total,
    COUNT(*) total_records,
    COUNT(invoice_total) number_of_invoices, 
    COUNT(payment_date) number_of_payments,  
    COUNT(DISTINCT client_id) number_of_distinct_clients
FROM invoices
WHERE invoice_date > '2019-07-01';
```

- `GROUP BY`配合聚合函数使用，进行分组统计
- 注意`WHERE`，`GROUP BY`，`ORDER BY`的先后顺序
- `GROUP BY`也可以多列组合为依据分组，逗号分隔即可

```sql
SELECT 
    client_id,  
    SUM(invoice_total) AS total_sales
FROM invoices
WHERE invoice_date >= '2019-07-01'
GROUP BY client_id
ORDER BY total_sales DESC;

SELECT 
    state,
    city,
    SUM(invoice_total) AS total_sales
FROM invoices
JOIN clients USING (client_id) 
GROUP BY state, city  
ORDER BY state;
```

- `HAVING`关键字对`SELECT`后的结果列进行事后筛选，通常用于分组聚合后查询
- `HAVING`和 `WHERE` 都是是条件筛选语句，条件写法相通
- `HAVING`使用结果列明，`WHERE`使用原表列名

```sql
SELECT 
    client_id,
    SUM(invoice_total) AS total_sales,
    COUNT(*/invoice_total/invoice_date) AS number_of_invoices
FROM invoices
GROUP BY client_id
HAVING total_sales > 500 AND number_of_invoices > 5;
```

- Mysql提供了`ROLLUP`运算符对聚合值进行汇总，如下将返回各客户的发票总额以及所有人的总发票额
- `ROLLUP`不是标准的SQL语言

```sql
SELECT 
    client_id,
    SUM(invoice_total)
FROM invoices
GROUP BY client_id WITH ROLLUP;
```

- 当`GROUP BY`取多列时，`ROLLUP`可以进行多层汇总
- 下面将返回各州的各市的发票总额以及州层次和全国层次的汇总额

```sql
SELECT 
    state,
    city,
    SUM(invoice_total) AS total_sales
FROM invoices
JOIN clients USING (client_id) 
GROUP BY state, city WITH ROLLUP;
```

-  使用`ROLLUP`时，`GROUP BY`不能使用列别名

```sql
SELECT 
    pm.name AS payment_method,
    SUM(amount) AS total
FROM payments p
JOIN payment_methods pm
    ON p.payment_method = pm.payment_method_id
GROUP BY pm.name WITH ROLLUP;
```

## 编写复杂查询

- Mysql将先执行括号内的子查询，之后将结果返回给外查询

```sql
SELECT *
FROM products
WHERE unit_price > (
    SELECT unit_price
    FROM products
    WHERE product_id = 3
);

SELECT *
FROM products
WHERE product_id NOT IN (
    SELECT DISTINCT product_id
    FROM order_items
);
```

- 子查询将一张表的查询结果作为另一张表的查询依据并层层嵌套；链接将这些表合并成一个包含所需全部信息的详情表再直接在详情表里筛选查询；两种方法一般是可互换的，具体用哪一种取决于效率和可读性
- 例：查找从未订购过的顾客，下面两种方法均可达到效果

```sql
SELECT *
FROM clients
WHERE client_id NOT IN (
    SELECT DISTINCT client_id
    FROM invoices
);

SELECT *
FROM clients
LEFT JOIN invoices USING (client_id)
WHERE invoices_id IS NULL;
```

- 例：选出买过生菜（id = 3）的顾客的id，姓和名，采用混合子查询+表连接的方法

```sql
SELECT customer_id, first_name, last_name
FROM customers
WHERE customer_id IN (  
    SELECT customer_id
    FROM orders
    JOIN order_items USING (order_id)  
    WHERE product_id = 3
);
```

- `ALL`关键字代表集合中的任意记录

```sql
SELECT *
FROM invoices
WHERE invoice_total > (
    SELECT MAX(invoice_total)
    FROM invoices
    WHERE client_id = 3
);

SELECT *
FROM invoices
WHERE invoice_total > ALL (
    SELECT invoice_total
    FROM invoices
    WHERE client_id = 3
);
```

- `ANY`关键字代表集合中存在某条记录，可与`MIN`等效

```sql
SELECT *
FROM invoices
WHERE invoice_total > ANY (
SELECT invoice_total
FROM invoices
WHERE client_id = 3
);

SELECT *
FROM invoices
WHERE invoice_total > (
SELECT MIN(invoice_total)
FROM invoices
WHERE client_id = 3
);
```

- `ANY`某些时候也可与`IN`等效
- 例：选择至少有两次发票记录的顾客

```sql
SELECT *
FROM clients
WHERE client_id IN (
    SELECT client_id
    FROM invoices
    GROUP BY client_id
    HAVING COUNT(*) >= 2
);

SELECT *
FROM clients
WHERE client_id = ANY ( 
    SELECT client_id
    FROM invoices
    GROUP BY client_id
    HAVING COUNT(*) >= 2
);
```

- 在上面的子查询中，子查询与主查询无关，可以先进行完子查询，再进行主查询
- 相关子查询，子查询与主查询有关
- 例：返回各部门中工资超过该部门平均值的人，对于`employees e`中的每条记录，都将执行一次子查询

```sql
SELECT *
FROM employees e  -- 关键 1
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE office_id = e.office_id  -- 关键 2
    -- 【子查询表字段不用加前缀，主查询表的字段要加前缀，以此区分】
);
```

- `EXISTS`关键字如果集合不为空返回`TRUE`，否则返回`FALSE`
- `IN + 子查询` 等效于 `EXIST + 相关子查询`，如前者子查询结果集过大占用内存，后者逐条验证更有效率

```sql
SELECT *
FROM clients
WHERE client_id IN (
    SELECT DISTINCT client_id
    FROM invoices
);

SELECT *
FROM clients c
WHERE EXISTS (
    SELECT *
    -- 就这个子查询的目的来说，SELECT的选择不影响结果，
    -- 因为EXISTS()函数只根据是否为空返回结果
    FROM invoices
    WHERE client_id = c.client_id
);
```

- `EXISTS()`返回的是 `TRUE/FALSE`，所以也可以加上`NOT`取反

```sql
SELECT *
FROM products 
WHERE product_id NOT IN (
    FROM order_items
);

SELECT *
FROM products p
WHERE NOT EXISTS (
    SELECT *
    FROM order_items
    WHERE product_id = p.product_id
);
```

- 例：经典题目，选择没有选所有课的学生

```sql
-- 选择没有选所有课的学生
-- 存在某一样课，该学生没有对应的选课记录
SELECT *
FROM stu s
WHERE EXISTS(
  	SELECT *
  	FROM class c
  	WHERE NOT EXISTS(
      	SELECT *
      	FROM stu_class
      	WHERE stu_id = s.stu_id AND class_id = c.class_id
    )
);
```

- 子查询或相关子查询也可用于`SELECT`中
- 引用同级的列别名，需要使用`SELECT`，引用同级的列别名不需要说明来源
- 下面的例子中有三个子查询，第一个为相关子查询，表示某顾客订单总量，第二个为子查询，表示所有订单平均值，第三个引用同级的列别名，表示前两者的差

```sql
SELECT 
    client_id,
    name,
    (SELECT SUM(invoice_total) FROM invoices WHERE client_id = c.client_id) AS total_sales,
    (SELECT AVG(invoice_total) FROM invoices) AS average,
    (SELECT total_sales - average) AS difference   
FROM clients c;
```

- 子查询也可用于`FROM`中
- 当子查询太复杂时应使用视图将子查询结果储存起来，使用视图作为来源表
- 在FROM中使用子查询，即使用 “派生表” 时，必须给派生表取个别名

```sql
SELECT * 
FROM (
    SELECT 
        client_id,
        name,
        (SELECT SUM(invoice_total) FROM invoices WHERE client_id = c.client_id) AS total_sales,
        (SELECT AVG(invoice_total) FROM invoices) AS average,
        (SELECT total_sales - average) AS difference   
    FROM clients c
) AS sales_summury
WHERE total_sales IS NOT NULL;
```

## 基本函数

内置的用来处理数值、文本、日期等的函数

- 数值函数

```sql
SELECT ROUND(5.7365, 2)  -- 四舍五入
SELECT TRUNCATE(5.7365, 2)  -- 截断
SELECT CEILING(5.2)  -- 天花板函数，大于等于此数的最小整数
SELECT FLOOR(5.6)  -- 地板函数，小于等于此数的最大整数
SELECT ABS(-5.2)  -- 绝对值
SELECT RAND()  -- 随机函数，0到1的随机值
```

- 字符串函数

```sql
SELECT LENGTH('sky')  -- 字符串字符个数

SELECT UPPER('sky')  -- 转大写
SELECT LOWER('Sky')  -- 转小写

SELECT LTRIM('  Sky')  -- 去除左多余空格
SELECT RTRIM('Sky  ')  -- 去除右多余空格
SELECT TRIM(' Sky ')  -- 去除多余空格

SELECT LEFT('Kindergarden', 4)  -- 取左4个字符
SELECT RIGHT('Kindergarden', 6)  -- 取右6个字符
SELECT SUBSTRING('Kindergarden', 7, 6)  -- 从第7个开始长度为6的子串，省略长度参数则取到末端

SELECT LOCATE('gar', 'Kindergarden')  -- 定位首次出现的位置，没有的话返回0，不区分大小写

SELECT REPLACE('Kindergarten', 'garten', 'garden')  -- 替换

SELECT CONCAT(f_name, ' ', l_name) AS f_name FROM customers  -- 连接
```

- 处理时间日期的函数

```sql
-- 当前时间日期对象
SELECT NOW()  -- 2021-08-27 08:50:46
SELECT CURDATE()  -- 2021-08-27
SELECT CURTIME()  -- 08:50:46

-- 提取时间日期对象中的元素
SELECT YEAR(NOW())  -- 2021

SELECT DAYNAME(NOW())  -- Saturday
SELECT MONTHNAME(NOW())  -- September
```

- 日期格式化，格式说明符里，大小写是不同的，这是目前SQL里第一次出现大小写不同的情况

```sql
SELECT DATE_FORMAT(NOW(), '%M %d, %Y')  -- September 12, 2020
SELECT TIME_FORMAT(NOW(), '%H:%i %p')  -- 11:07 AM
```

## SELECT 语句执行顺序

其中每一个操作都会产生一张虚拟的表，这个虚拟的表作为一个处理的输入，只是这些虚拟的表对用户来说是透明的，只有最后一个虚拟的表才会被作为结果返回。 

1. FORM：对表计算笛卡尔积，产生虚表VT1。
2. ON：对虚表VT1进行ON筛选，只有那些符合`join-condition`的行才会被记录在虚表VT2中。
3. JOIN：如果指定了OUTER JOIN（比如left join、 right join），那么保留表中未匹配的行就会作为外部行添加到虚拟表VT2中，产生虚拟表VT3, 如果from子句中包含两个以上的表的话，那么就会对上一个join连接产生的结果VT3和下一个表重复执行步骤1~3这三个步骤，一直到处理完所有的表为止。
4. WHERE：对虚拟表VT3进行WHERE条件过滤。只有符合`where-condition`的记录才会被插入到虚拟表VT4中。
5. GROUP BY：根据group by子句中的列，对VT4中的记录进行分组操作，产生VT5。
6. HAVING：对虚拟表VT5应用having过滤，只有符合`having-condition`的记录才会被插入到虚拟表VT6中。
7. SELECT：执行select操作，选择指定的列，插入到虚拟表VT7中。
8. DISTINCT：对VT7中的记录进行去重。产生虚拟表VT8。
9. ORDER BY: 将虚拟表VT8中的记录按照<order_by_list>进行排序操作，产生虚拟表VT9。
10. LIMIT：取出指定行的记录，产生虚拟表VT10,并将结果返回。

## 参考资料

[伟大的海伦君.SQL进阶教程](https://www.bilibili.com/video/BV1UE41147KC?p=34)

[ASC2050.Mosh完全掌握SQL笔记](https://zhuanlan.zhihu.com/p/222865842)

