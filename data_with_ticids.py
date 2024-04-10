import numpy as np 
import pandas as pd 
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

df_complete = pd.read_csv('/mnt/sdceph/users/rzhang/BDD_MM_IACOB_output_contemp_RZhang_C+SB1.csv')
df_inc = pd.read_csv('/mnt/sdceph/users/rzhang/BDD_MM_IACOB_output_contemp_RZhang_SB2+Oe+Mag.csv')

def get_tic_id(star_name):
    """
    Given the name used in this data, see if there is an alternative name for it in the TIC catalog
    by quering the SIMBAD astronomical database
    """
    enc_star_name = quote(star_name)
    search_url = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={enc_star_name}&submit=SIMBAD+search"
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        element = text.find('TIC')
        if element != -1:
            tic_id = text[element+4:element+13]
            return tic_id 
        else:
            print(f"TIC ID not found for {star_name}")
            return np.nan
    else:
        print(f"Failed to retrieve data for {star_name}")
        return np.nan

df_complete['TIC_ID'] = df_complete['ID'].apply(get_tic_id)
df_complete['TIC_ID'] = df_complete['TIC_ID'].str.replace('\n', '')

df_complete.to_csv('/mnt/sdceph/users/rzhang/iacob1.csv')

df_inc['TIC_ID'] = df_inc['ID'].apply(get_tic_id)
df_inc['TIC_ID'] = df_inc['TIC_ID'].str.replace('\n', '')

df_inc.to_csv('/mnt/sdceph/users/rzhang/iacob2.csv')
