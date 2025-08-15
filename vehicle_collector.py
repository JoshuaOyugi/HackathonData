import requests
import json
import time
import logging
from datetime import datetime
import csv
import os
import pandas as pd
import zipfile
import numpy as np
import random
from typing import Dict, List, Optional

class KenyaVehicleCollector:
    def __init__(self, start_year: int = 2000, end_year: int = 2019):
        self.base_url = "https://vpic.nhtsa.dot.gov/api"
        self.start_year = start_year
        self.end_year = end_year
        self.session = requests.Session()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'kenya_vehicles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create data storage directory
        self.data_dir = "kenya_vehicle_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data storage
        self.all_vehicle_data = []
        self.epa_data = None
        self.fuel_type_cache = {}
        
        # Initialize Kenya-specific mappings
        self.init_kenya_mappings()
        
        # Load EPA data
        self.load_epa_data()
    
    def init_kenya_mappings(self):
        """Initialize Kenya market mappings"""
        
        # Country of manufacture mapping (realistic for Kenya market)
        self.country_mapping = {
            'TOYOTA': 'Japan',
            'NISSAN': 'Japan', 
            'HONDA': 'Japan',
            'MAZDA': 'Japan',
            'MITSUBISHI': 'Japan',
            'SUBARU': 'Japan',
            'ISUZU': 'Japan',
            'SUZUKI': 'Japan',
            'HYUNDAI': 'South Korea',
            'KIA': 'South Korea',
            'BMW': 'Germany',
            'MERCEDES-BENZ': 'Germany',
            'AUDI': 'Germany',
            'VOLKSWAGEN': 'Germany',
            'FORD': 'USA',
            'CHEVROLET': 'USA',
            'PEUGEOT': 'France',
            'RENAULT': 'France',
            'LAND ROVER': 'UK',
            'JAGUAR': 'UK',
            'VOLVO': 'Sweden'
        }
        
        # Service cost base rates by engine capacity (KES per year)
        self.service_cost_by_capacity = {
            (0.0, 1.2): {'min': 45000, 'max': 85000},      # Small engines
            (1.2, 1.6): {'min': 60000, 'max': 120000},     # Compact engines  
            (1.6, 2.0): {'min': 80000, 'max': 150000},     # Mid-size engines
            (2.0, 2.5): {'min': 120000, 'max': 220000},    # Large engines
            (2.5, 3.5): {'min': 180000, 'max': 350000},    # Performance engines
            (3.5, 10.0): {'min': 250000, 'max': 500000}    # Heavy-duty engines
        }
        
        # Brand multipliers for service costs
        self.brand_multipliers = {
            'TOYOTA': 0.9,      # Reliable, cheaper parts
            'NISSAN': 0.95,
            'HONDA': 0.9,
            'MAZDA': 1.0,
            'MITSUBISHI': 1.0,
            'SUBARU': 1.1,
            'HYUNDAI': 0.85,
            'KIA': 0.85,
            'BMW': 1.8,         # Expensive parts/service
            'MERCEDES-BENZ': 1.9,
            'AUDI': 1.7,
            'VOLKSWAGEN': 1.3,
            'FORD': 1.1,
            'CHEVROLET': 1.1,
            'PEUGEOT': 1.4,
            'LAND ROVER': 2.2,
            'JAGUAR': 2.0
        }
        
        # Age multipliers (older cars cost more to service)
        self.age_multipliers = {
            (0, 3): 0.8,    # New cars
            (3, 7): 1.0,    # Mid-age
            (7, 12): 1.3,   # Older
            (12, 20): 1.6,  # Very old
            (20, 50): 2.0   # Vintage
        }
    
    def safe_convert_value(self, value):
        """Convert pandas/numpy types to JSON-serializable Python types"""
        if pd.isna(value):
            return 'Unknown'
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value) if not np.isnan(value) else 'Unknown'
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return str(value) if value is not None else 'Unknown'
    
    def load_epa_data(self):
        """Load EPA fuel economy data"""
        self.logger.info("Loading EPA fuel economy data...")
        
        csv_files = [f for f in os.listdir(self.data_dir) if 'vehicles.csv' in f]
        
        if csv_files:
            try:
                csv_path = os.path.join(self.data_dir, csv_files[0])
                self.epa_data = pd.read_csv(csv_path)
                self.logger.info(f"Loaded EPA data: {len(self.epa_data)} records")
                return
            except Exception as e:
                self.logger.error(f"Error loading EPA file: {e}")
        
        self.download_epa_data()
    
    def download_epa_data(self):
        """Download EPA fuel economy data"""
        self.logger.info("Downloading EPA database...")
        
        try:
            epa_url = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
            response = self.session.get(epa_url, timeout=120)
            
            if response.status_code == 200:
                zip_path = os.path.join(self.data_dir, "vehicles.csv.zip")
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                csv_path = os.path.join(self.data_dir, "vehicles.csv")
                self.epa_data = pd.read_csv(csv_path)
                self.logger.info(f"Successfully loaded EPA data: {len(self.epa_data)} vehicles")
            else:
                self.logger.error(f"Failed to download EPA data: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error downloading EPA data: {e}")
    
    def get_fuel_economy_data(self, make: str, model: str, year: int) -> Dict:
        """Get fuel economy data from EPA database with smart fallbacks"""
        if self.epa_data is None:
            return self.get_smart_estimates(make, model, year)
        
        try:
            cache_key = f"{make.lower()}_{model.lower()}_{year}"
            if cache_key in self.fuel_type_cache:
                return self.fuel_type_cache[cache_key]
            
            # Filter EPA data
            mask = (
                (self.epa_data['year'] == year) &
                (self.epa_data['make'].str.contains(make, case=False, na=False))
            )
            
            # Try exact model match
            model_matches = self.epa_data[mask & 
                self.epa_data['model'].str.contains(model, case=False, na=False)]
            
            # Try partial match
            if len(model_matches) == 0:
                model_first = model.split()[0] if model else ""
                if model_first:
                    model_matches = self.epa_data[mask & 
                        self.epa_data['model'].str.contains(model_first, case=False, na=False)]
            
            # Use make/year only
            if len(model_matches) == 0:
                model_matches = self.epa_data[mask]
            
            if len(model_matches) > 0:
                vehicle = model_matches.iloc[0]
                
                fuel_data = {
                    'fuel_type': self.safe_convert_value(vehicle.get('fuelType', 'Petrol')),
                    'engine_capacity': self.safe_convert_value(vehicle.get('displ', 1.8)),
                    'combined_mpg': self.safe_convert_value(vehicle.get('comb08', 25)),
                    'body_type': self.safe_convert_value(vehicle.get('VClass', 'Sedan'))
                }
                
                # Calculate mileage (km/L) and consumption (L/100km)
                try:
                    mpg = float(fuel_data['combined_mpg'])
                    if mpg > 0:
                        fuel_data['mileage_kmpl'] = round(mpg * 0.425144, 1)  # MPG to km/L
                        fuel_data['consumption_l100km'] = round(235.214 / mpg, 1)
                    else:
                        fuel_data['mileage_kmpl'] = 12.0
                        fuel_data['consumption_l100km'] = 8.3
                except:
                    fuel_data['mileage_kmpl'] = 12.0
                    fuel_data['consumption_l100km'] = 8.3
                
                self.fuel_type_cache[cache_key] = fuel_data
                return fuel_data
            
            return self.get_smart_estimates(make, model, year)
            
        except Exception as e:
            self.logger.error(f"Error getting fuel data for {make} {model} {year}: {e}")
            return self.get_smart_estimates(make, model, year)
    
    def get_smart_estimates(self, make: str, model: str, year: int) -> Dict:
        """Smart estimates based on make/model patterns"""
        make_upper = make.upper()
        model_lower = model.lower()
        
        # Default values
        fuel_type = 'Petrol'
        engine_capacity = 1.8
        mileage = 12.0
        body_type = 'Sedan'
        
        # Engine capacity estimation
        if any(x in model_lower for x in ['vitz', 'march', 'note', 'fit', 'swift']):
            engine_capacity = random.uniform(1.0, 1.3)
            mileage = random.uniform(14.0, 18.0)
            body_type = 'Hatchback'
        elif any(x in model_lower for x in ['corolla', 'civic', 'axio', 'belta']):
            engine_capacity = random.uniform(1.3, 1.6)
            mileage = random.uniform(12.0, 16.0)
            body_type = 'Sedan'
        elif any(x in model_lower for x in ['fielder', 'wingroad', 'caldina']):
            engine_capacity = random.uniform(1.5, 1.8)
            mileage = random.uniform(11.0, 14.0)
            body_type = 'Station Wagon'
        elif any(x in model_lower for x in ['rav4', 'crv', 'x-trail', 'forester']):
            engine_capacity = random.uniform(2.0, 2.5)
            mileage = random.uniform(9.0, 12.0)
            body_type = 'SUV'
        elif any(x in model_lower for x in ['prado', 'pajero', 'land cruiser']):
            engine_capacity = random.uniform(2.7, 4.0)
            mileage = random.uniform(6.0, 9.0)
            body_type = 'SUV'
        elif any(x in model_lower for x in ['hilux', 'ranger', 'triton']):
            engine_capacity = random.uniform(2.5, 3.0)
            mileage = random.uniform(8.0, 11.0)
            body_type = 'Pickup'
        elif any(x in model_lower for x in ['hiace', 'serena', 'stepwgn']):
            engine_capacity = random.uniform(2.0, 2.7)
            mileage = random.uniform(8.0, 12.0)
            body_type = 'Van'
        
        # Brand-specific adjustments
        if make_upper in ['BMW', 'MERCEDES-BENZ', 'AUDI']:
            engine_capacity = max(engine_capacity, 2.0)
            mileage *= 0.8  # Luxury cars typically less efficient
        elif make_upper == 'TOYOTA':
            mileage *= 1.1  # Toyota known for efficiency
        
        # Hybrid detection
        if any(x in model_lower for x in ['prius', 'hybrid', 'insight']):
            fuel_type = 'Hybrid'
            mileage = random.uniform(18.0, 25.0)
        
        # Diesel detection
        if any(x in model_lower for x in ['d4d', 'tdi', 'diesel']):
            fuel_type = 'Diesel'
            mileage *= 1.2  # Diesel more efficient
        
        consumption = round(100 / mileage, 1)
        
        return {
            'fuel_type': fuel_type,
            'engine_capacity': round(engine_capacity, 1),
            'mileage_kmpl': round(mileage, 1),
            'consumption_l100km': consumption,
            'body_type': body_type
        }
    
    def estimate_service_cost(self, make: str, engine_capacity: float, year: int) -> int:
        """Estimate annual service cost in KES"""
        current_year = datetime.now().year
        age = current_year - year
        
        # Get base cost by engine capacity
        base_cost = 120000  # Default
        
        try:
            capacity = float(engine_capacity)
            for (min_cap, max_cap), cost_range in self.service_cost_by_capacity.items():
                if min_cap <= capacity < max_cap:
                    base_cost = random.randint(cost_range['min'], cost_range['max'])
                    break
        except:
            base_cost = 120000
        
        # Apply brand multiplier
        brand_multiplier = self.brand_multipliers.get(make.upper(), 1.0)
        base_cost *= brand_multiplier
        
        # Apply age multiplier
        age_multiplier = 1.0
        for (min_age, max_age), multiplier in self.age_multipliers.items():
            if min_age <= age < max_age:
                age_multiplier = multiplier
                break
        
        base_cost *= age_multiplier
        
        # Add some randomness to make it realistic
        variation = random.uniform(0.85, 1.15)
        final_cost = int(base_cost * variation)
        
        # Round to nearest 5000 for realism
        return round(final_cost / 5000) * 5000
    
    def get_country_of_manufacture(self, make: str, year: int) -> str:
        """Get country of manufacture"""
        make_upper = make.upper()
        
        # Special case: Toyota has local assembly in Kenya for recent years
        if make_upper == 'TOYOTA' and year >= 2018:
            return random.choice(['Japan', 'Kenya (Assembly)'])
        
        return self.country_mapping.get(make_upper, 'Japan')
    
    def make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            time.sleep(0.5)  # Rate limiting
            return data
            
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return None
    
    def get_all_makes(self) -> List[Dict]:
        """Get all vehicle makes"""
        self.logger.info("Fetching vehicle makes...")
        
        data = self.make_api_request("vehicles/GetMakesForVehicleType/car?format=json")
        if data and 'Results' in data:
            all_makes = data['Results']
            
            valid_makes = []
            for make in all_makes:
                make_name = make.get('MakeName', '').strip()
                make_id = make.get('MakeId', '')
                
                if make_name and make_id:
                    valid_makes.append({
                        'Make_Name': make_name,
                        'Make_ID': make_id
                    })
            
            self.logger.info(f"Found {len(valid_makes)} vehicle makes")
            return valid_makes
        return []
    
    def get_models_for_make_year(self, make_name: str, year: int) -> List[Dict]:
        """Get models for specific make and year"""
        make_clean = make_name.replace(' ', '%20').replace('&', '%26')
        endpoint = f"vehicles/GetModelsForMakeYear/make/{make_clean}/modelyear/{year}?format=json"
        data = self.make_api_request(endpoint)
        
        if data and 'Results' in data:
            models = data['Results']
            standardized = []
            for model in models:
                standardized.append({
                    'Model_Name': model.get('Model_Name', model.get('ModelName', '')),
                    'Model_ID': model.get('Model_ID', model.get('ModelId', '')),
                    'Make_Name': model.get('Make_Name', model.get('MakeName', make_name)),
                    'Make_ID': model.get('Make_ID', model.get('MakeId', ''))
                })
            return standardized
        return []
    
    def collect_kenya_vehicles(self, makes_limit: int = 20, years_limit: int = 19):
        """Collect Kenya-specific vehicle data - MASSIVE DATASET"""
        self.logger.info("🇰🇪 Starting MASSIVE Kenya vehicle data collection...")
        self.logger.info(f"📊 Target: 6000+ vehicles from {self.start_year}-{self.end_year}")
        
        all_makes = self.get_all_makes()
        if not all_makes:
            self.logger.error("Failed to fetch makes")
            return
        
        # Expand to more makes for massive dataset
        kenya_makes = [
            'TOYOTA', 'NISSAN', 'HONDA', 'MAZDA', 'MITSUBISHI', 'SUBARU', 
            'BMW', 'MERCEDES-BENZ', 'HYUNDAI', 'KIA', 'FORD', 'VOLKSWAGEN',
            'AUDI', 'CHEVROLET', 'PEUGEOT', 'RENAULT', 'VOLVO', 'LAND ROVER',
            'JAGUAR', 'LEXUS', 'INFINITI', 'ACURA', 'ISUZU', 'SUZUKI'
        ]
        
        filtered_makes = [m for m in all_makes if m.get('Make_Name') in kenya_makes]
        if not filtered_makes:
            filtered_makes = all_makes[:makes_limit]
        else:
            # Use ALL matching makes for massive dataset
            filtered_makes = filtered_makes[:makes_limit]
        
        # Process ALL years from 2000-2019
        years = list(range(self.start_year, min(self.start_year + years_limit, self.end_year + 1)))
        
        self.logger.info(f"🚀 Processing {len(filtered_makes)} makes across {len(years)} years")
        self.logger.info(f"📈 Expected combinations: {len(filtered_makes)} × {len(years)} = {len(filtered_makes) * len(years)} potential make-year pairs")
        
        total_processed = 0
        total_collected = 0
        
        for make_info in filtered_makes:
            make_name = make_info.get('Make_Name', '').strip()
            if not make_name:
                continue
            
            self.logger.info(f"🔧 Processing: {make_name}")
            make_start_time = time.time()
            make_vehicles = 0
            
            for year in years:
                year_start_time = time.time()
                models = self.get_models_for_make_year(make_name, year)
                total_processed += 1
                
                if not models:
                    self.logger.debug(f"  📅 {year}: No models found")
                    continue
                
                year_vehicles = 0
                
                for model_info in models:
                    model_name = model_info.get('Model_Name', '').strip()
                    if not model_name:
                        continue
                    
                    # Get vehicle data
                    fuel_data = self.get_fuel_economy_data(make_name, model_name, year)
                    
                    engine_capacity = fuel_data.get('engine_capacity', 1.8)
                    service_cost = self.estimate_service_cost(make_name, engine_capacity, year)
                    country = self.get_country_of_manufacture(make_name, year)
                    
                    # Create Kenya vehicle record
                    vehicle_record = {
                        'Make': make_name,
                        'Model': model_name,
                        'Fuel type': fuel_data.get('fuel_type', 'Petrol'),
                        'Capacity': f"{engine_capacity}L",
                        'YoM': year,
                        'Mileage': f"{fuel_data.get('mileage_kmpl', 12.0)} km/L",
                        'Consumption': f"{fuel_data.get('consumption_l100km', 8.3)} L/100km",
                        'Country of manufacture': country,
                        'Body type': fuel_data.get('body_type', 'Sedan'),
                        'Service cost': f"KES {service_cost:,}/year"
                    }
                    
                    self.all_vehicle_data.append(vehicle_record)
                    year_vehicles += 1
                    total_collected += 1
                    
                    # Progress logging every 100 vehicles
                    if total_collected % 100 == 0:
                        self.logger.info(f"    📊 Progress: {total_collected} vehicles collected so far...")
                
                year_time = time.time() - year_start_time
                if year_vehicles > 0:
                    self.logger.info(f"  📅 {year}: {year_vehicles} models collected ({year_time:.1f}s)")
                
                make_vehicles += year_vehicles
            
            make_time = time.time() - make_start_time
            self.logger.info(f"✅ {make_name} complete: {make_vehicles} vehicles ({make_time:.1f}s)")
            
            # Show running totals
            if total_collected >= 1000:
                self.logger.info(f"🎯 MILESTONE: {total_collected} vehicles collected!")
        
        self.logger.info(f"🎉 MASSIVE COLLECTION COMPLETE!")
        self.logger.info(f"📊 Total vehicles: {len(self.all_vehicle_data)}")
        self.logger.info(f"📈 Make-year combinations processed: {total_processed}")
        self.logger.info(f"✨ Target achieved: {'YES' if len(self.all_vehicle_data) >= 6000 else 'NO'} (6000+ goal)")
        
        # Show breakdown by make
        make_counts = {}
        for vehicle in self.all_vehicle_data:
            make = vehicle['Make']
            make_counts[make] = make_counts.get(make, 0) + 1
        
        self.logger.info(f"\n📋 Vehicles by make:")
        for make, count in sorted(make_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {make}: {count} vehicles")
    
    def save_kenya_csv(self):
        """Save Kenya vehicle data as clean CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"{self.data_dir}/kenya_vehicles_{timestamp}.csv"
        
        if not self.all_vehicle_data:
            self.logger.error("No data to save")
            return
        
        # Define exact column order
        columns = [
            'Make', 'Model', 'Fuel type', 'Capacity', 'YoM',
            'Mileage', 'Consumption', 'Country of manufacture', 
            'Body type', 'Service cost'
        ]
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.all_vehicle_data)
            
            self.logger.info(f"📊 Saved: {csv_file}")
            self.logger.info(f"📈 Records: {len(self.all_vehicle_data)}")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
    
    def print_summary(self):
        """Print detailed collection summary for massive dataset"""
        if not self.all_vehicle_data:
            return
        
        makes = set(v['Make'] for v in self.all_vehicle_data)
        years = set(v['YoM'] for v in self.all_vehicle_data)
        fuel_types = set(v['Fuel type'] for v in self.all_vehicle_data)
        body_types = set(v['Body type'] for v in self.all_vehicle_data)
        countries = set(v['Country of manufacture'] for v in self.all_vehicle_data)
        
        print("\n" + "="*70)
        print("🇰🇪 MASSIVE KENYA VEHICLE COLLECTION SUMMARY")
        print("="*70)
        print(f"📊 Total vehicles: {len(self.all_vehicle_data):,}")
        print(f"🎯 Target achieved: {'✅ YES' if len(self.all_vehicle_data) >= 6000 else '❌ NO'} (6000+ goal)")
        print(f"🏭 Makes: {len(makes)} - {', '.join(sorted(makes))}")
        print(f"📅 Years: {min(years)}-{max(years)} ({len(years)} years)")
        print(f"⛽ Fuel types: {', '.join(sorted(fuel_types))}")
        print(f"🚗 Body types: {', '.join(sorted(body_types))}")
        print(f"🌍 Countries: {', '.join(sorted(countries))}")
        
        # Top makes by count
        make_counts = {}
        for v in self.all_vehicle_data:
            make = v['Make']
            make_counts[make] = make_counts.get(make, 0) + 1
        
        print(f"\n🏆 Top 10 makes by vehicle count:")
        for i, (make, count) in enumerate(sorted(make_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1:2d}. {make}: {count:,} vehicles")
        
        # Year distribution
        year_counts = {}
        for v in self.all_vehicle_data:
            year = v['YoM']
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"\n📅 Vehicles by decade:")
        decade_2000s = sum(count for year, count in year_counts.items() if 2000 <= year <= 2009)
        decade_2010s = sum(count for year, count in year_counts.items() if 2010 <= year <= 2019)
        print(f"  2000-2009: {decade_2000s:,} vehicles")
        print(f"  2010-2019: {decade_2010s:,} vehicles")
        
        # Service cost analysis
        service_costs = []
        for v in self.all_vehicle_data:
            cost_str = v['Service cost'].replace('KES ', '').replace(',', '').replace('/year', '')
            try:
                cost = int(cost_str)
                service_costs.append(cost)
            except:
                pass
        
        if service_costs:
            avg_cost = sum(service_costs) / len(service_costs)
            min_cost = min(service_costs)
            max_cost = max(service_costs)
            
            print(f"\n💰 Service cost analysis (KES/year):")
            print(f"  Average: KES {avg_cost:,.0f}")
            print(f"  Range: KES {min_cost:,.0f} - KES {max_cost:,.0f}")
            print(f"  Budget cars (<100k): {sum(1 for c in service_costs if c < 100000):,} vehicles")
            print(f"  Mid-range (100k-200k): {sum(1 for c in service_costs if 100000 <= c < 200000):,} vehicles")
            print(f"  Premium (200k+): {sum(1 for c in service_costs if c >= 200000):,} vehicles")
        
        print(f"\n📋 Sample vehicles from massive dataset:")
        sample_indices = [0, len(self.all_vehicle_data)//4, len(self.all_vehicle_data)//2, 
                         len(self.all_vehicle_data)*3//4, len(self.all_vehicle_data)-1]
        
        for i, idx in enumerate(sample_indices):
            if idx < len(self.all_vehicle_data):
                vehicle = self.all_vehicle_data[idx]
                print(f"{i+1}. {vehicle['YoM']} {vehicle['Make']} {vehicle['Model']}")
                print(f"   Engine: {vehicle['Capacity']} {vehicle['Fuel type']} | Body: {vehicle['Body type']}")
                print(f"   Efficiency: {vehicle['Mileage']} | {vehicle['Consumption']}")
                print(f"   Service: {vehicle['Service cost']} | Origin: {vehicle['Country of manufacture']}")
        
        print(f"\n🗂️  Data saved to CSV with {len(self.all_vehicle_data):,} rows and 10 columns")


def main():
    """Main function - Generate MASSIVE Kenya vehicle dataset"""
    print("🇰🇪 MASSIVE Kenya Vehicle Data Collector")
    print("="*50)
    print("📊 Target: 6,000+ vehicles from 2000-2019")
    print("🚗 Same models across different years included")
    print("📈 Smart estimation for Kenya market")
    print("🔧 All popular makes and extensive coverage")
    print("="*50)
    
    # Initialize for massive collection
    collector = KenyaVehicleCollector(start_year=2000, end_year=2019)
    
    start_time = time.time()
    
    try:
        print("🚀 Starting massive data collection...")
        print("⏱️  This will take several minutes due to API calls...")
        
        # Collect MASSIVE Kenya vehicle dataset
        collector.collect_kenya_vehicles(makes_limit=25, years_limit=20)
        
        # Save as CSV
        collector.save_kenya_csv()
        
        # Print detailed summary
        collector.print_summary()
        
        total_time = time.time() - start_time
        vehicles_count = len(collector.all_vehicle_data)
        
        print(f"\n🎉 MASSIVE COLLECTION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Vehicles collected: {vehicles_count:,}")
        print(f"⚡ Rate: {vehicles_count/total_time:.1f} vehicles/second")
        print(f"📁 Files saved in: {collector.data_dir}/")
        print(f"🎯 Goal achieved: {'YES! 🎉' if vehicles_count >= 6000 else f'NO - Got {vehicles_count}, need 6000+'}")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        print("💾 Saving partial data...")
        collector.save_kenya_csv()
        collector.print_summary()
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        # Save whatever we collected
        if collector.all_vehicle_data:
            collector.save_kenya_csv()
            print(f"💾 Saved partial data: {len(collector.all_vehicle_data)} vehicles")


if __name__ == "__main__":
    main()