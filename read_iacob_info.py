import numpy as np 
import pandas as pd 
import requests
from bs4 import BeautifulSoup


df_complete = pd.read_csv('/mnt/sdceph/users/rzhang/BDD_MM_IACOB_output_contemp_RZhang_C+SB1.csv')
df_inc = pd.read_csv('/mnt/sdceph/users/rzhang/BDD_MM_IACOB_output_contemp_RZhang_SB2+Oe+Mag.csv')

print('ID', df_complete['ID'])

def get_tic_id(star_name):
    """
    Given the name used in this data, see if there is an alternative name for it in the TIC catalog
    by quering the SIMBAD astronomical database
    """
    search_url = "http://simbad.u-strasbg.fr/simbad/sim-fbasic"
    
    # Send a GET request to SIMBAD with the search parameters
    response = requests.get(search_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        print('response.text', response.text)
        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        print('soup', soup)
        # Search for the TIC ID in the parsed HTML
        # Note: You may need to adjust the search parameters depending on the actual HTML structure of the page
        tic_id_element = soup.find(string='TIC')
        print('tic_id_element', tic_id_element)
        if tic_id_element:
            # If the TIC ID is found, extract and return it
            # This part may need customization based on the page's structure around the TIC ID
            tic_id = tic_id_element.find_next().text
            print('tic id', tic_id)
            return tic_id
        else:
            print(f"TIC ID not found for {star_name}")
            return None
    else:
        print(f"Failed to retrieve data for {star_name}")
        return None

star_name = "HD 74920"
tic_id = get_tic_id(star_name)
if tic_id:
    print(f"The TIC ID for {star_name} is: {tic_id}")
else:
    print("Could not find a TIC ID.")

   
    