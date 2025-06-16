import os
import sys
import numpy as np
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import math
# import time
import pyptf.smooth_data as sd


class ScenarioPlayer:
    """
    Version: 0.1.31
    A class for parsing the scenario data, including GNSS, earthquake, and sea level data.
    When initialising, you must specify the paths to the data files (eq_path, sl_path, gnss_path)
    or define them in the next section - then the class can be initiated without arguments.
    """
    def __init__(self, **kwargs):

        """
        Initializes the ScenarioPlayer class with the specified paths.

        Args:
            data_path (str): Path to the data archive
        """
        data_path = kwargs.get('data_path', None)

        self.sl_path = os.path.join(data_path, 'service_sl.json')
        self.gnss_path = os.path.join(data_path, 'GNSS_data.npy')  # Path to the GNSS data -To be fixed

        self.search_radius_sld = 2500. #km

    def run_scenario_player(self, event_dict):

        # Sea Level Data 
        close_services = self.get_closest_services(event_dict['lat'], event_dict['lon'], self.search_radius_sld)

        origin_time = event_dict['ot']
        # Convert origin_time to datetime object
        datetime_object = datetime.strptime(origin_time, '%Y-%m-%dT%H:%M:%S')
        # Calculate the time range
        dt1 = datetime_object - timedelta(hours=12)
        dt2 = datetime_object + timedelta(hours=12)
        time_range = (dt1.strftime('%Y-%m-%dT%H:%M:%S'), dt2.strftime('%Y-%m-%dT%H:%M:%S'))

        stations_with_data = []
        stations_without_data = []
        sealevel_dict = dict()
        for station in close_services:
            ssc_id = station.get('ssc_id', 'N/A').replace("SSC-","")  # Get 'ssc_id', default to 'N/A' if not found; remove "SSC-"
            name = station.get('name', 'N/A')      # Get 'name', default to 'N/A' if not found
            # get data
            data_avl, sealevel_df = self.fetch_sl_data(origin_time, time_range, ssc_id)

            # append station information to the appropriate list depending on the data availability
            if data_avl:
                sealevel_df = sealevel_df.loc[sealevel_df['sec'] >= 0]
                max_stat = sealevel_df['sea_level'].max()
                # if not np.isnan(max_stat): 
                try:
                    print(f"Station: {name} | Max {max_stat} m")
                    tide = sd.smooth(sec = sealevel_df['sec'].to_numpy(), sealevel = sealevel_df['sea_level'].to_numpy())
                    sealevel_df['sea_level'] = sealevel_df['sea_level'] - tide
                
                    sealevel_dict[name] = dict()
                    sealevel_dict[name]['coords'] = (float(station['geo:lat']), float(station['geo:lon']))
                    sealevel_dict[name]['data'] = sealevel_df
                    stations_with_data.append({"ssc_id": ssc_id, "name": name})
                # else:
                except:
                    # TODO Valparaiso station: problem with the radar sensor (prs would work)
                    stations_without_data.append({"ssc_id": ssc_id, "name": name, "lon": float(station['geo:lon']), "lat": float(station['geo:lat'])})
            else:
                stations_without_data.append({"ssc_id": ssc_id, "name": name, "lon": float(station['geo:lon']), "lat": float(station['geo:lat'])})


        # print the results
        print(f"Stations with data available: {len(stations_with_data)}")
        for station in stations_with_data:
            print(f"  ssc_id: {station['ssc_id']}, name: {station['name']}")
        print(f"Stations without data available: {len(stations_without_data)}")
        # for station in stations_without_data:
        #     print(f"  ssc_id: {station['ssc_id']}, name: {station['name']}")

        # GNSS Data
        data_gnss = self.fetch_gnss_data() # get the GNSS data as dictionary
        # Convert the dictionary to DataFrame
        gnss_df = pd.DataFrame.from_dict(data_gnss)
        print(f"Number of GNSS stations with data: {len(gnss_df)}")
        # print(gnss_df.to_string()) #SITE Lat Lon An(m) Sn Ae(m) Se Av(m) Sv
    
        return sealevel_dict, gnss_df

    def get_closest_services(self, reference_lat, reference_lon, radius):
        """
        Finds sea level stations within a specified radius.
        """
        def haversine(lat1, lon1, lat2, lon2):
            """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
            """
            # convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371 # Radius of earth in kilometers.
            return c * r

        try:
            with open(self.sl_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {self.sl_path}")
            exit()
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in the file.")
            exit()

        close_services = []
        search_radius = radius  # in kilometers

        for record in data:
            try:
                service_lat = float(record.get('geo:lat', 0))  # Handle missing keys
                service_lon = float(record.get('geo:lon', 0))

                distance = haversine(reference_lat, reference_lon, service_lat, service_lon)

                if distance < search_radius:
                    close_services.append(record)
            except (ValueError, TypeError):
                print(f"Warning: Skipping record due to invalid latitude or longitude: {record}")

        print(f"Number of SLD stations within {search_radius} km: {len(close_services)}")
        return close_services


    def fetch_sl_data(self, origin_time, time_range, station_code):
        """
        Fetches sea level data for a certain station from a web API.
        url = "https://www.ioc-sealevelmonitoring.org/service.php"
        Note: the data is not filtered (original data from a sensor).

        Args:
            time_range (tuple): Start and end times for data retrieval.
            station_code (str): Codename of the station.

        Returns:
            data (tuple): The sea level data from the API.
            data_avl (bool): True if data is available, False otherwise.
            df (pd.DataFrame): A DataFrame containing sea level data
            relative to the time of the event (origin_time) in seconds.
            The origin_time is defined by earthquake parameters.
            None: If an error occurs.

        """

        params = {
            "query": "data",
            "format": "json",
            "code": station_code,
            # "includesensors[]": "prs",
            "timestart": time_range[0],
            "timestop": time_range[1]
        }

        url = "http://www.ioc-sealevelmonitoring.org/service.php"  # Base URL

        try:
            response = requests.get(url, params=params, verify=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data: {e}")
            data_avl = False
            df = None
            return data_avl, df

        try:
            data = json.loads(response.text)
            data_avl = True
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON data: {e}")
            data_avl = False
            df = None
            return data_avl, df

        if not data:
            # print(f"No data available for the specified time range (station: {params['code']}).")
            data_avl = False
            df = None
            return data_avl, df

        dates = [item["stime"] for item in data]
        slevels = [item["slevel"] for item in data]

        # Convert dates to seconds since origin_time
        # origin_time = self.fetch_eq_param()['ot']
        seconds_since_eq = [self.calculate_seconds_diff(origin_time, d) for d in dates]

        dates = pd.to_datetime(dates)
        df = pd.DataFrame({"date": dates, "sec": seconds_since_eq, "sea_level": slevels})

        # return data, data_avl, df
        return data_avl, df


    def fetch_gnss_data(self):
        """
        Reads and returns GNSS data from an .npy file.

        Args:
            None
        Returns:
            dict: The GNSS data.
            None: If an error occurs during file reading.
        """
        try:
            gnss = np.load(self.gnss_path, allow_pickle=True).item()
            return gnss
        except FileNotFoundError:
            print(f"WARNING: GNSS data file not found at {self.gnss_path}")
            return None
        except Exception as e:
            print(f"Error reading GNSS data: {e}")
            return None


    def calculate_seconds_diff(self, time1: str, time2: str) -> int:
        """
        Calculate the time difference in seconds between two timestamps.

        Parameters:
        time1,time2 (str): Can be an ISO 8601 string (like "2015-10-23T19:02:29+00:00","2015-10-23 19:02:29" ) or just a date (e.g., "2015-10-23").

        Returns:
        int: The difference in seconds between the two timestamps.

        Raises:
        TypeError: If unable to process naive and aware datetime properly.
        ValueError: If timestamps are in an invalid format.
        """

        # Parse the input time strings, assuming they are in ISO 8601 format
        try:
              # Parse the datetimes
              dt1 = datetime.fromisoformat(time1.strip()) #remove posible leading and tailing spaces
              dt2 = datetime.fromisoformat(time2.strip()) #remove posible leading and tailing spaces

              # Make both datetimes offset-aware if they're not already
              if dt1.tzinfo is None:
                  dt1 = dt1.replace(tzinfo=timezone.utc)  # Assume UTC if no timezone in time1
              if dt2.tzinfo is None:
                  dt2 = dt2.replace(tzinfo=timezone.utc)  # Assume UTC if no timezone in time2

              diff = dt2 - dt1

              return int(diff.total_seconds())

        except TypeError as e:
            raise TypeError("Ensure both datetime objects are offset-aware or offset-naive.") from e
        except ValueError as e:
            raise ValueError("Ensure the input timestamps are in valid ISO 8601 format.") from e

    # def fetch_eq_param(self):
    #     """
    #     Reads earthquake parameters from an .npy file.

    #     Args:
    #         None
    #     Returns:
    #         dict: Earthquake parameters.
    #         None: If an error occurs.
    #     """
    #     try:
    #         ev = np.load(self.eq_path, allow_pickle=True).item()
    #         return ev
    #     except FileNotFoundError:
    #         print(f"Error: Earthquake data file not found at {self.eq_path}")
    #         return None
    #     except Exception as e:
    #         print(f"Error reading earthquake data: {e}")
    #         return None

