'''Module used to extract and/or combine astronomical data from Sloan Digital Sky Survey.'''
from astropy.table import Table
from astroquery.sdss import SDSS
import pandas as pd

class Data:
    '''Retrieves and stores SDSS data within class.'''

    def __init__(self):
        self.data = pd.DataFrame()
        self.spectrum = []
    
    def extract_from_query(self, query:str):
        '''
        Queries from SDSS with user-specified query.
        Stores pandas DataFrame in 'data' attribute.

        Parameters
        ----------
        query : str
            Full query written in ADQL.
        '''

        query_result = SDSS.query_sql(query)
        self.data = query_result.to_pandas()

    def extract_from_constraints(self, constraints:dict):
        '''
        Queries data from SDSS with user-given constraints.  
        Stores pandas DataFrame in 'data' attribute.

        Parameters
        ----------
        contraints : dict
            Dictionary that specifies the requested SDSS data.

            Must resemble:
            {
            "database" : table_name,
            "constrain" : [ ["column1", [ lower_bound1, upper_bound1] ],
                            ["column2", [ None,         upper_bound2] ],
                            ["column3", [ lower_bound3,       None] ]... ]
            }

        Raises
        ------
        ValueError
            Raised if more than one database is specified in the constraints.
        '''

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
        '''Reads and stores data from csv file in 'data' attribute. Assumes header.'''

        self.data = pd.read_csv(path, skiprows=1)

    @staticmethod
    def get_spectra_from_obj_id(obj_id:str):
        '''
        Queries data from SDSS with user-given constraints.  
        Stores pandas DataFrame in 'data' attribute.

        Parameters
        ----------
        obj_id = a string corresponding to an object ID

        Returns
        -------

        return the spectrum linked to the obj_id

        Raises
        ------
        ValueError
            Raised if obj_id is not a string
        '''
        if type(obj_id)!=str:
            raise ValueError("obj_id should be a string")
        query = rf"SELECT * FROM SpecObj where specObjID in ({obj_id})"
        query_result = SDSS.query_sql(query)
        return SDSS.get_spectra(matches=query_result)
    
    def get_spectra_from_data(self):
        '''
        Queries data from SDSS with user-given constraints.  
        Stores pandas DataFrame in 'data' attribute.

        Returns
        -------

        return the spectrum linked to the Data.data elements ids

        Raises
        ------
        ValueError
            Raised if Data.data is None
        '''
        if self.data is None or "specObjID" not in self.data:
            raise ValueError("self.data should contain \"specObjID\" ")
        obj_ids= tuple(self.data["specObjID"].apply(lambda x : str(x)))
        query=rf"SELECT TOP 10 * FROM SpecObj where specObjID in {obj_ids}"
        query_result = SDSS.query_sql(query)
        self.spectrum= SDSS.get_spectra(matches=query_result)
        return self.spectrum

    def write_file(self, path:str):
        '''Writes contents of 'data' attribute to csv file.'''

        self.data.to_csv(path, index=False)

    def concat(self, new_data, axis:int):
        '''
        Concatenates new queried data to the existing data stored in class.

        Parameters
        ----------
        new_data : Data class object
            Contains new data in 'data' attribute.

        axis : int, either 0 or 1
        '''

        self.data = pd.concat([self.data, new_data.data], axis=0)

    def merge(self, new_data, on_column:str):
        '''
        Performs inner join between existing and new data.

        Parameters
        ----------
        new_data : Data class object
            Contains new data in 'data' attribute.

        on_column : str
            Column for inner join.
        '''

        left = self.data
        right = new_data.data
        self.data = left.merge(right, how="inner", on=on_column)