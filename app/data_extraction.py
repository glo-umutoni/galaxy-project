''' '''
from astroquery.sdss import SDSS
import pandas as pd

class Data:
    ''' '''
    def __init__(self):
        self.data = pd.DataFrame()
    
    def extract_from_query(self, query:str):
        query_result = SDSS.query_sql(query)
        self.data = query_result.to_pandas()

    def extract_from_constraints(self, constraints:dict):
        if len(constraints["database"]) > 1:
            raise ValueError("try again. do better. only one.... smh")
        
        constrain_list = []
        col_constrains = constraints["constrain"]
        for col in col_constrains:
            values = col[1]
            lower = values[0]
            upper = values[1]
            col_name = col[0]

            if lower == upper:
                if (type(lower) == str): constrain_list.append(f"({col_name} = \'{lower}\')")
                else: constrain_list.append(f"({col_name} = {lower})")
            elif lower is None: constrain_list.append(f"({col_name} < {upper})")
            elif upper is None: constrain_list.append(f"({col_name} > {lower})")
            else: constrain_list.append(f"({col_name} BETWEEN {lower} AND {upper})")

        print(constrain_list)
        query_constrain = f"WHERE {' AND '.join(constrain_list)}"
        print(query_constrain)
        query = f"SELECT {', '.join(constraints['columns'])} FROM {', '.join(constraints['database'])} " + query_constrain
        print(query)
        query_result = SDSS.query_sql(query)
        self.data = query_result.to_pandas()
        

    def extract_from_file(self, path:str):
        self.data = pd.read_csv(path, skiprows=1)

    def write_file(self, path:str):
        self.data.to_csv(path, index=False)

    def concat(self, new_data, axis:int):
        self.data = pd.concat([self.data, new_data.data], axis=0)

    def merge(self, new_data, on_column:str):
        left = self.data
        right = new_data.data
        self.data = left.merge(right, how="inner", on=on_column)