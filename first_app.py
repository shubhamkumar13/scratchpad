from re import U
import streamlit as st
import numpy as np
import pandas as pd

from nested_dict import nested_dict
from pprint import pprint

import uuid
import json
import os
import pandas as pd
import pandas.io.json as pdjson
import seaborn as sns
from collections import namedtuple

st.title("Sequential Benchmarks")

artifacts_dir = "/home/sk/sandmark-nightly/sequential"

bench_files = []

# Problem : right now the structure is a nested dict of
#     `(hostname * (timestamp * (variants list)) dict ) dict`
# and this nested structure although works but it is a bit difficult to work with
# so we need to create a class object which is a record type and add functions to it

# <host 1>
# |--- <timestamp 1>
#         |--- <commit 1>
#                 |--- <variant 1>
#                 |--- <variant 2>
#                 ....
#                 ....
#                 ....
#                 |--- <variant n>
#         |--- <commit 2>
#                 ....
#                 ....
#                 ....
#         ....
#         ....
#         ....
#         |--- <commit n>
#                 ....
#                 ....
#                 ....
# ....
# ....
# ....
# ....
# |--- <timestamp n>
#         ....
#         ....
# <host 2>
# ....
# ....
# ....
# <host n>

class StupidIdea:
    def __init__(self):
        self.structure = nested_dict(3, list)
    
    def add(self, host, timestamp, commit, variant):
        self.structure[host][timestamp][commit].append(variant)
    
    def __repr__(self):
        return f'{self.structure}'


x = StupidIdea()
st.write(x)


# Loads file metadata
for root, dirs, files in os.walk(artifacts_dir):
    for file in files:
        if file.endswith("_1.orun.summary.bench"):
            f = root.split("/sequential")
            bench_files.append((os.path.join(root,file)))

def files_to_dict(files):
    benches = nested_dict(2, list)
    for x in files:
        l = x.split("/sequential/")[1]
        d = l.split("/")
        host         = d[0]
        timestamp    = d[1]
        commit       = d[2]
        variant      = d[3]
        variant_root = d[3].split('_')[0]
        variant_stem = d[3].split('_')[1]
        value        = variant_root + '+' + commit + '_' + variant_stem
        benches[host][timestamp].append(value)
    benches = dict(benches)
    for i in benches.items():
        benches[i[0]] = dict(sorted(i[1].items(), key=lambda t : t[0], reverse=True))
    return benches

benches = files_to_dict(bench_files)

st.write(benches)
st.write(bench_files)

n = int(st.text_input('Number of benchmarks','2'))

containers = [st.columns(3) for i in range(n)]

def get_selected_values(n):
    lst = []
    for i in range(n):
        # create the selectbox in columns
        host_val = containers[i][0].selectbox('hostname', benches.keys(), key = str(i) + '0')
        timestamp_val = containers[i][1].selectbox('timestamp', benches[host_val], key = str(i) + '1')
        variant_val = containers[i][2].selectbox('variant', benches[host_val][timestamp_val], key = str(i) + '2')
        lst.append({"host" : host_val, "timestamp" : timestamp_val, "variant" : variant_val})
    return lst

selected_file_list = get_selected_values(n)

# Expander for showing bench files
with st.expander("Show metadata for selected benchmarks"):
    st.write(selected_file_list)

def get_filepath(file):
    host_val, timestamp_val, variant_val = file[0], file[1], file[2]

    # create file name
    commit = variant_val.split('_')[0].split('+')[-1]
    variant_stem = variant_val.split('_')[1]
    variant_value = '+'.join(variant_val.split('_')[0].split('+')[:-1]) + '_' + variant_stem

    file = os.path.join(
        artifacts_dir,
        host_val,
        timestamp_val,
        commit,
        variant_value
    )

    return file

def get_dataframe_from_file(file):
    # json to dataframe
    data_frames = []

    with open(file) as f:
        data = []
        for l in f:
            data.append(json.loads(l))
        df = pdjson.json_normalize(data)
        value     = file.split('/sequential/')[1]
        date      = value.split('/')[1].split('_')[0]
        commit_id = value.split('/')[2][:7]
        variant   = value.split('/')[3].split('_')[0]
        df["variant"] = variant + '_' + date + '_' + commit_id
        data_frames.append(df)

    df = pd.concat (data_frames, sort=False)
    df = df.sort_values(['name'])
    return df

selected_file_path = [get_filepath(list(file.values())) for file in selected_file_list]
df_list = [get_dataframe_from_file(file) for file in selected_file_path]

with st.expander("Show table for the selected benchmarks"):
    for i in df_list:
        st.write(i)

st.header("Select baseline")
base_container = st.columns(3)
baseline_h, baseline_t, baseline_v = (
    base_container[0].selectbox(
        'hostname', 
        [file["host"] for file in selected_file_list],
        key = str(uuid.uuid4())    
    ),
    base_container[1].selectbox(
        'timestamp', 
        [file["timestamp"] for file in selected_file_list],
        key = str(uuid.uuid4())    
    ),
    base_container[2].selectbox(
        'variant', 
        [file["variant"] for file in selected_file_list],
        key = str(uuid.uuid4())    
    ))