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
    search_url = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={star_name}&submit=SIMBAD+search"
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        element = text.find('TIC')
        if element:
            tic_id = text[element+4:element+13]
            return tic_id 
        else:
            print(f"TIC ID not found for {star_name}")
            return None
    else:
        print(f"Failed to retrieve data for {star_name}")
        return None

# star_name = "HD74920"
# tic_id = get_tic_id(star_name)
# if tic_id:
#     print(f"The TIC ID for {star_name} is: {tic_id}")
# else:
#     print("Could not find a TIC ID.")

df_complete['TIC_ID'] = df_complete['ID'].apply(get_tic_id)
df_complete['TIC_ID'] = df_complete['TIC_ID'].str.replace('\n', '')
print(df_complete.filter(['ID', 'TIC_ID']))