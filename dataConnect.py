import csv

import pandas as pd
import pymysql
from sqlalchemy import create_engine

# 设置数据库连接参数
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'db': 'test',
    'charset': 'utf8mb4',  # 推荐使用utf8mb4编码
    'port': 3306  # MySQL的默认端口是3306
}
conn = pymysql.connect(**db_config)
print("已连接")


# 尝试连接数据库
def write_db(name, path):
    try:
        # 读取CSV文件
        df = pd.read_csv(path, encoding='utf-8-sig')
        if df is not None:
            # 创建SQLAlchemy引擎
            engine = create_engine('mysql+pymysql://{user}:{password}@{host}:{port}/{db}'.format(**db_config))
            # 将DataFrame存储到SQL数据库中
            df.to_sql(f'{name}', con=engine, if_exists='replace', index=False)
            conn.commit()
        else:
            print("读取CSV文件失败，请检查文件路径和文件编码。")
    except pymysql.MySQLError as e:
        print("数据库连接错误：", e)
    finally:
        # 关闭数据库连接
        if conn is not None:
            conn.close()


def read(csv_file_name, sqlname):
    try:
        with conn.cursor() as cursor:
            # 执行SQL查询
            cursor.execute(f"SELECT * FROM {sqlname}")  #
            result = cursor.fetchall()

            # 创建CSV文件并写入数据
            with open(csv_file_name, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                # 写入标题行（即列名）
                header = [column[0] for column in cursor.description]
                csv_writer.writerow(header)

                # 写入查询结果
                for row in result:
                    csv_writer.writerow(row)

    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        # 关闭数据库连接
        conn.close()

    print(f"数据已导出到CSV文件：{csv_file_name}")


def fetch(colname, sqlname):
    query = "SELECT {} FROM {};".format(colname, sqlname)

    Attributes_counts = {}
    try:
        # 建立游标并执行查询
        cursor = conn.cursor()
        cursor.execute(query)
        # 获取所有结果
        Attributes = cursor.fetchall()
        print("attributes:",Attributes)
        # 将元组列表转换为普通列表
        Attributes_list = [Attribute[0] for Attribute in Attributes]

        # 统计各个属性的总数
        for Attribute in Attributes_list:
            Attributes_counts[Attribute] = Attributes_counts.get(Attribute, 0) + 1

        print(Attributes_counts)

    except Exception as e:
        print("Error while fetching data:", e)
    return Attributes_counts


def submit_conn(indices, correlation_matrix_values):
    cursor = conn.cursor()

    # 创建submit_data表
    cursor.execute('''CREATE TABLE IF NOT EXISTS submit_data ({});'''.format(
        ", ".join([f"`{index}` VARCHAR(255)" for index in indices])))

    # 插入indices数据
    if indices:
        cursor.execute(
            '''INSERT INTO submit_data ({}) VALUES ({});'''.format(", ".join(indices),
                                                                   ", ".join(['%s' for _ in indices])),
            indices)
    # 插入correlation_matrix_values数据
    for row in correlation_matrix_values:
        cursor.execute('''INSERT INTO submit_data ({}) VALUES ({});'''.format(", ".join(indices),
                                                                              ", ".join(['%s' for _ in indices])), row)

    # 提交更改
    conn.commit()

    # 关闭游标和连接
    cursor.close()
    conn.close()


def fetch_submit(tablename):
    # 连接到数据库
    # 获取数据库游标
    print("ok")
    cursor = conn.cursor()

    # 执行查询语句，获取所有数据
    cursor.execute("SELECT * FROM {}".format(tablename))  # 使用字符串格式化来插入表名

    # 获取列名
    #index = [col[0] for col in cursor.description]
    #print(index)
    # 初始化存储数据的列表
    values = []

    # 逐行读取数据
    row = cursor.fetchone()
    while row is not None:
        # 将元组转换为列表
        row_list = list(row)
        values.append(row_list)  # 将列表形式的数据存储到values列表中
        row = cursor.fetchone()

    # 清空表内容
    # 删除表 submit_data（包括其结构和所有数据）
    cursor.execute("DROP TABLE {}".format(tablename))

    # 提交事务
    conn.commit()
    # 关闭游标和连接
    cursor.close()
    conn.close()

    # 打印列名和数据
    print("数据：", values)

#fetch('年龄','total_sql')