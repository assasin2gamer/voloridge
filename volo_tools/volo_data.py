import importlib
import numpy as np
import pandas as pd
from volo_tools import volo_feature
importlib.reload(volo_feature)

'''
Initial data processing
2:40am 11/05/2025

'''

class VoloDataProcessor:
    def __init__(self):
        self.data = None

    def process(self, data_path):
        ''' 
        CSV has: Code,Symbol,Date,Close,Volume,Adjustment Factor
        Seperate by Symbol
        '''
        self.data = pd.read_csv(data_path)
        # Sort data by Symbol and Date
        self.data.sort_values(by=['Symbol', 'Date'], inplace=True)
        # Replace price with adjusted price
        self.data['Adj_Close'] = self.data['Close'] * self.data['Adjustment Factor']
        
        # Print stocks with adjustment factor not equal to 1 and how many times it changes
        adjustment_issues = self.data[self.data['Adjustment Factor'] != 1]
        if not adjustment_issues.empty:
            adjustment_counts = adjustment_issues.groupby('Symbol').size()
            print("Stocks with adjustment factor changes:")
            print(adjustment_counts)
        
        # Create a csv for each stock symbol
        symbols = self.data['Symbol'].unique()
        for symbol in symbols:
            symbol_data = self.data[self.data['Symbol'] == symbol]
            # sort by date (month/day/year)
            symbol_data['Date'] = pd.to_datetime(symbol_data['Date'], format='%m/%d/%Y')
            symbol_data.sort_values(by='Date', inplace=True)
            
            
            symbol_data.to_csv(f"data/{symbol}_data.csv", index=False)
            
        # Range of dates for each stock and save as a csv for sanity check
        date_ranges = []
        for symbol in symbols:
            symbol_data = self.data[self.data['Symbol'] == symbol]
            start_date = symbol_data['Date'].min()
            end_date = symbol_data['Date'].max()
            date_ranges.append({'Symbol': symbol, 'Start Date': start_date, 'End Date': end_date})
        date_ranges_df = pd.DataFrame(date_ranges)
        date_ranges_df.to_csv("data/date_ranges.csv", index=False)

        result = volo_feature.VoloFeatureProcessor(self.data).feature_process()
        
        
        return result
            
    
    
        
    
