import requests
import json

def debug_nhtsa_api():
    """Debug what NHTSA API is actually returning"""
    print("🔍 Debugging NHTSA API Response...")
    print("="*50)
    
    try:
        url = "https://vpic.nhtsa.dot.gov/api/vehicles/GetMakesForVehicleType/car?format=json"
        print(f"📡 Making request to: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"✅ Status Code: {response.status_code}")
        
        data = response.json()
        print(f"📊 Total results: {data.get('Count', 0)}")
        
        if 'Results' in data:
            results = data['Results']
            print(f"🔢 Results array length: {len(results)}")
            
            # Examine first 10 results
            print("\n📋 First 10 raw results:")
            for i, make in enumerate(results[:10]):
                print(f"{i+1}. Raw data: {make}")
                make_name = make.get('Make_Name', '')
                make_id = make.get('Make_ID', '')
                print(f"   Make_Name: '{make_name}' (length: {len(str(make_name))})")
                print(f"   Make_ID: '{make_id}' (type: {type(make_id)})")
                print(f"   Has valid name: {bool(make_name and str(make_name).strip())}")
                print(f"   Has valid ID: {bool(make_id)}")
                print()
            
            # Check data types and common issues
            print("🔍 Data Analysis:")
            empty_names = 0
            empty_ids = 0
            valid_entries = 0
            
            for make in results:
                make_name = make.get('Make_Name', '')
                make_id = make.get('Make_ID', '')
                
                if not make_name or not str(make_name).strip():
                    empty_names += 1
                if not make_id:
                    empty_ids += 1
                if make_name and str(make_name).strip() and make_id:
                    valid_entries += 1
            
            print(f"Empty names: {empty_names}")
            print(f"Empty IDs: {empty_ids}")
            print(f"Valid entries: {valid_entries}")
            
            # Show some valid entries
            print("\n✅ First 5 valid entries:")
            count = 0
            for make in results:
                make_name = make.get('Make_Name', '')
                make_id = make.get('Make_ID', '')
                
                if make_name and str(make_name).strip() and make_id:
                    print(f"{count+1}. {make_name} (ID: {make_id})")
                    count += 1
                    if count >= 5:
                        break
            
        else:
            print("❌ No 'Results' key in response")
            print("Response keys:", list(data.keys()))
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    debug_nhtsa_api()
    