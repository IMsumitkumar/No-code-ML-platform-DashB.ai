import mysql.connector as connection
import pandas as pd

class Data_From_MySQL:
    def __init__(self, host, username, password, database=None, table_name=None):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.table_name = table_name
    
    def connect_mydb(self):
        # making a connection between database and python
        my_db = connection.connect(host=self.host, username=self.username,
                                    password=self.password, database = self.database, use_pure=True)
        print(my_db.is_connected())
        
        # this will select all data from table 
        # make sure to give database name
        query = "select * from "+self.table_name+";"

        result_dataFrame = pd.read_sql(query, my_db)

        my_db.close()
        return result_dataFrame

# getting data from MongoDB (database)
class Data_From_MongoDB:
    def __init__(self):
        pass

    def connect_mydb(self):
        pass


def DB_from_servers(host, username, password, database, table_name=None,
                    connect_to_mysql=False):
    if connect_to_mysql==True:
        x = Data_From_MySQL(host=host, username=username, password=password, database=database, table_name=table_name)
        return x.connect_mydb()
    else:
        print("True or False")


    # if connect_to_mongo==True:
    #     x = Data_From_MongoDB()
    #     return x.connect_mydb()
    # else: