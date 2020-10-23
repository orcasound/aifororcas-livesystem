import os, sys, json, glob
import requests, bs4, urllib
import tqdm
import pandas as pd
from os.path import dirname
from urllib.parse import urljoin

class TqdmUpTo(tqdm.tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def _geturlsoup(url):
    response = requests.get(url)
    return bs4.BeautifulSoup(response.text,"html.parser")

def _get_table_rows(url,occurence=0):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text,"html.parser")
    return soup.findAll('table')[occurence].findAll('tr')

def _get_urls_from_select_button(soup,baseurl):
    url_dict = {}
    options = soup.findAll('option')
    for option in options:
        key = option.string.strip().lower()
        value = '/'.join([baseurl,option['value']])
        url_dict[key]=value
    if "select" in url_dict: del url_dict["select"]
    return url_dict

def parse_database_page_to_tsv(url,tsvfile):
    baseurl = '/'.join(url.split('/',3)[:-1])
    rows = _get_table_rows(url)
    getstr = lambda x: x.string.strip() if x.string is not None else ""

    with open(tsvfile,'w') as f:
        f.write("name\tlocation\tdate\taudio\tmetadata\n")
        for row in rows[1:]: # skip title row
            contents = row.findAll('td')
            name, loc, date, audio, metadata = getstr(contents[1]), \
                getstr(contents[2]),\
                getstr(contents[3]),\
                urljoin(baseurl,contents[4].a['href']),\
                '/'.join([
                    url.rsplit('/',1)[0],
                    contents[5].a['href'].split('(')[1].split(')')[0].replace('\'','')
                ])
            f.write('\t'.join([name,loc,date,audio,metadata])+'\n')
    print("Parsed links for",len(rows),"rows to file:",tsvfile)

def select_and_get_urls(main_url,common_name): # for fun could be done recursively
    main_soup = _geturlsoup(main_url)
    select_buttons = main_soup.findAll('select')
    # select name get initial page url
    url_dict = _get_urls_from_select_button(select_buttons[0],dirname(main_url))
    select_soup = _geturlsoup(url_dict[common_name])
    select_buttons = select_soup.findAll('select')
    return _get_urls_from_select_button(select_buttons[2],dirname(main_url))

def get_metadata(mtd_url):
    metadata = {}
    rows = _get_table_rows(mtd_url,1)
    for row in rows[1:]:
        contents = row.findAll('td')
        metadata[contents[0].string.replace(':','')] = contents[1].string
    return metadata

def where_are_the_whales(main_url,save_dir):
    soup = _geturlsoup(main_url).findAll('select')[0] # main select drop down
    ud = _get_urls_from_select_button(soup,dirname(main_url))
    common_names = [ k for k in ud.keys() if 'whale' in k ] # just get all whale species

    for common_name in common_names:
        url_years = select_and_get_urls(main_url,common_name)
        filedir = os.path.join(save_dir,common_name)
        os.makedirs(filedir,exist_ok=True)
        for year, url in url_years.items():
            parse_database_page_to_tsv(
                url,
                "{}_{}.tsv".format(os.path.join(filedir,common_name),str(year))
                )

num_lines = lambda x: sum(1 for line in open(x))
def fetch_all_metadata(save_dir,whale):
    whales = os.listdir(save_dir)

    metadata_json = {}
    for tsv_file in glob.glob(os.path.join(save_dir,whale,"*.tsv")):
        with open(tsv_file,'r') as f:
            next(f) # skip header row
            for line in tqdm.tqdm(f,total=num_lines(tsv_file)):
                mtd_url = line.split('\t')[-1].strip()
                metadata_json[mtd_url] = get_metadata(mtd_url)

    with open(os.path.join(save_dir,whale,"metadata.json"),'w') as f:
        json.dump(metadata_json,f)

def download_from_url(dl_url,dl_dir):
    # download only if not already exists
    file_name = os.path.basename(dl_url)
    dl_path = os.path.join(dl_dir,file_name)
    if os.path.isfile(dl_path):
        print("Skipping",file_name,"as it already exists.")
    else:
        print("Downloading",file_name)
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc=dl_url.split('/')[-1]) as t:  # all optional kwargs
            urllib.request.urlretrieve(dl_url, filename=dl_path,
                            reporthook=t.update_to)

def download_all_cuts(save_dir,whale,wav_dir):
    cuts_tsv = os.path.join(save_dir,whale,"allcuts.tsv")
    wav_dir = os.path.join("./data/wavcut",whale)

    os.makedirs(wav_dir,exist_ok=True)
    df = pd.read_csv(cuts_tsv,sep='\t')

    # get file 
    for dl_url in tqdm.tqdm(df.audio):
        download_from_url(dl_url,wav_dir)


if __name__ == "__main__":
    main_url = "https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm"
    save_dir = "./data/tsv"
    whale = "killer whale"
    cuts_tsv = os.path.join(save_dir,whale,"allcuts.tsv")
    wav_dir = os.path.join("./data/wavcut",whale)
