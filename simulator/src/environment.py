import random
import pandas as pd
import os

class Weather:
    """
    Simulates environmental conditions.
    Now acts as a Data-Driven Predictive Digital Twin component, 
    pulling real weather data from the Cleaned dataset.
    """

    def __init__(self, config_data):
        """
        Initializes the weather model.

        Args:
            config_data (dict): The 'simulation' section from simulation_config.yaml.
        """
        self.season = config_data.get('season', 'Summer')
        
        # Load the real dataset!
        try:
            # We look for the file in the root directory (one level up from src)
            # or in the current execution directory.
            csv_path = 'Cleaned_Solar_Weather_Dataset.csv'
            if not os.path.exists(csv_path):
                # Fallback path if running from inside src
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Cleaned_Solar_Weather_Dataset.csv')
                
            self.data = pd.read_csv(csv_path)
            self.max_steps = len(self.data)
        except Exception as e:
            raise RuntimeError(f"Could not load Cleaned_Solar_Weather_Dataset.csv. Please ensure it exists.") from e
        
        self.current_step_index = 0

    def get_weather_conditions(self):
        """
        Pulls the real weather conditions for the current time step.
        
        Returns:
            dict: Current Temperature_C, Humidity_percent, Irradiance_Wm2
        """
        # Get row and loop around if simulation runs longer than dataset
        row = self.data.iloc[self.current_step_index % self.max_steps]
        
        conditions = {
            'Temperature_C': float(row['Temperature_C']),
            'Humidity_percent': float(row['Humidity_percent']),
            'Irradiance_Wm2': float(row['Irradiance_Wm2'])
        }
        
        self.current_step_index += 1
        return conditions


class HomeLoad:
    """
    Represents the 'AC Home Load' from the diagram.
    
    Simulates household energy consumption with a base load and 
    stochastic spikes during peak hours.
    """

    def __init__(self, config_data):
        """
        Initializes the load profile.

        Args:
            config_data (dict): The 'load' section from simulation_config.yaml.
        """
        self.base_load_kw = config_data.get('base_load_kw', 0.5)
        self.peak_load_kw = config_data.get('peak_load_kw', 3.0)
        self.peak_start = config_data.get('peak_start_hour', 18) # 6 PM
        self.peak_end = config_data.get('peak_end_hour', 21)     # 9 PM

    def get_current_load(self, hour_of_day):
        """
        Calculates the instantaneous power demand of the house.
        
        Args:
            hour_of_day (float): Current hour (0-23).
            
        Returns:
            float: Power demand in kW.
        """
        # 1. Start with the base load (Fridge, Router, etc.)
        current_load = self.base_load_kw

        # 2. Add random noise/spikes
        # Random spikes up to 3 kW during peak hours (6-9 PM)"
        if self.peak_start <= hour_of_day <= self.peak_end:
            # Higher probability of high spikes during peak time
            spike = random.uniform(0, self.peak_load_kw)
        else:
            # Lower spikes during off-peak (occasional usage)
            spike = random.uniform(0, self.peak_load_kw * 0.1) # 10% of peak

        return current_load + spike


class UtilityGrid:
    """
    Represents the 'Utility Grid' from the diagram.
    
    Defines the constraints and costs for importing/exporting energy.
    """

    def __init__(self, config_data):
        """
        Initializes grid parameters.

        Args:
            config_data (dict): The 'grid' section from simulation_config.yaml.
        """
        self.export_limit_kw = config_data.get('export_limit_kw', 20.0)
        self.cost_import = config_data.get('cost_import_cents', 0.75)
        self.price_export = config_data.get('price_export_cents', 0.90)

    def calculate_cost(self, imported_kwh, exported_kwh):
        """
        Calculates the financial balance for a time period.
        
        Args:
            imported_kwh (float): Energy bought from grid.
            exported_kwh (float): Energy sold to grid.
            
        Returns:
            float: Net cost (positive = you pay, negative = you earn).
        """
        cost = imported_kwh * self.cost_import
        earnings = exported_kwh * self.price_export
        return cost - earnings