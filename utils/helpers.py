import os
import os.path as osp
import pandas as pd
from datetime import datetime
import re, ast

DIR = os.getcwd()

TOKEN = ''
GerritAccount = ""
XSRF_TOKEN = ""

def convert(seconds):
    # Calculate days, hours, minutes, and remaining seconds
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return f'{days:.2f} days, {hours:.2f} hours, {minutes:.2f} minutes and {seconds:.2f} seconds'


def generate_date(header):
    """Generate a date with a passed-in title
    """
    date = datetime.now().strftime("%Y/%m/%d %H:%M:%S:%f")
    return date, "{} {}".format(header, date)


def diff_dates(start_date, end_date):
    """Calculate the difference between two dates
    """
    start_date = datetime.strptime(start_date, "%Y/%m/%d %H:%M:%S:%f")
    end_date = datetime.strptime(end_date, "%Y/%m/%d %H:%M:%S:%f")

    elapsed_seconds = convert((end_date - start_date).seconds)

    print("This script took {}".format(elapsed_seconds))

def list_file(directory):
    """List files of a directory
    """
    files = [f for f in os.listdir(directory)]
    return files

def flatten_list(array):
    '''Flattens array items
    Example: [[1], [2, 3], [4]] becomes [1, 2, 3, 4]
    '''
    result = [item for sublist in array for item in sublist]
    return result

def combine_openstack_data(revert_path=False, filter_merged=True, filter_real_dev=True):
    '''Combine generated csv files into a single DataFrame object
    '''
    print('Reading OpenStack changes...')
    data_path = f"{os.getcwd()}"

    if revert_path == True:
        data_path += "/../Changes/"
    else:
        data_path += "/Changes/"


    df = pd.DataFrame([])
    changes_file_names = list_file(data_path)
    for f in changes_file_names:
        df_per_file = pd.read_csv(f"{data_path}{f}")
        df = pd.concat((df, df_per_file))

    # if filter_merged == True:
    #     df = df[(df['status'] == 'MERGED')]

    # if filter_real_dev == True:
    #     df = df[df['is_owner_bot'] == False]
    
    df = df.drop_duplicates(subset=["number"])

    df = df.sort_values(by="created", ascending=True).reset_index(drop=True)

    df['created'] = df['created'].map(lambda x: x[:-10])

    df["created"] = pd.to_datetime(df["created"])

    df.loc[df['project'].str.startswith('openstack/'), 'project'] = df['project'].map(lambda x: x[10:])

    print(f'OpenStack changes loaded successfully...')
    
    return df

def time_diff(start, end):
    """Compute the time difference between two dates in seconds
    """
    if start > end:
        start, end = end, start
    current_date =  datetime.strptime(end, "%Y-%m-%d %H:%M:%S") 
    previous_date = datetime.strptime(start, "%Y-%m-%d %H:%M:%S") 
    diff = current_date - previous_date
    diff = float("{:.2f}".format(diff.total_seconds() / 3600))
    return diff

def to_date_format(date):
    """Compute the time difference between two dates in seconds
    """
    return datetime.strftime(date, "%Y-%m-%d %H:%M:%S") 
    

def convert_seconds_to_years(seconds):
    """Convert seconds into years
    """
    conversion_factor = 365.25 * 24 * 60 * 60
    years = seconds / conversion_factor
    return years

def preprocess_change_description(text):
    # Pattern to match
    pattern_1 = r"(Depends-On|Needed-By)(:\s[a-zA-Z0-9/\.\:\+\-\#]{1,})"

    # Replacement string
    replacement = " "

    # Perform the replacement using re.sub()
    depends_needed_new_text = re.sub(pattern_1, replacement, text)

    # # Pattern to match
    pattern_2 = r"(Related-Bug:\s#\d+)"

    # Perform the replacement using re.sub()
    new_related_bug_text = re.sub(pattern_2, replacement, depends_needed_new_text)

    # Pattern to match
    pattern_3 = r"Change-Id:\s[a-zA-Z0-9]{41}"

    # Perform the replacement using re.sub()
    new_change_id_text = re.sub(pattern_3, replacement, new_related_bug_text)

    # Pattern to match
    pattern_4 = r"\.[\n|\t]{1}"

    # Perform the replacement using re.sub()
    new_sanititzed_text = re.sub(pattern_4, ". ", new_change_id_text)

    # Pattern to match
    pattern_5 = r"[\n|\t]"

    # Perform the replacement using re.sub()
    new_text = re.sub(pattern_5, replacement, new_sanititzed_text).strip()

    return new_text


def combine_changed_file_names(x):
    tokens = []
    for cf in ast.literal_eval(x):
        tokens += re.split('/|_', cf)

    return tokens