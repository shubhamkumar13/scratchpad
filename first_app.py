from re import U, split
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
from itertools import chain

st.title("Sequential Benchmarks")

artifacts_dir = "/home/sk/sandmark-nightly/sequential"

bench_files = []

# Problem : right now the structure is a nested dict of
#     `(hostname * (timestamp * (variants list)) dict ) dict`
# and this nested structure although works but it is a bit difficult to work with
# so we need to create a class object which is a record type and add functions to 

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

#This idea is only for sandmark nightly
class StupidIdea:
    config = {}

    def __init__(self):
        self.structure = nested_dict(3, list)
        self.config["bench_type"] = "/sequential/"
        self.config["artifacts_dir"] = "/home/sk/sandmark-nightly/sequential"
    
    def add(self, host, timestamp, commit, variant):
        self.structure[host][timestamp][commit].append(variant)
    
    def add_files(self, files):
        for x in files:
            l = x.split(self.config["bench_type"])[1]
            d = l.split("/")
            self.add(
                d[0],
                d[1],
                d[2],
                d[3]
            )
    
    def to_filepath(self):
        lst = []
        for host, timestamps in self.structure.items():
            for timestamp, commits in timestamps.items():
                for commit, variant_list in commits.items():
                    file = os.path.join(
                        self.config["artifacts_dir"],
                        self.config["/sequential/"],
                        host,
                        timestamp,
                        commit,
                        variant_list[0]
                    )
                    lst.append(file)
        return lst


    def __repr__(self):
        return f'{self.structure}'


# Loads file metadata
for root, dirs, files in os.walk(artifacts_dir):
    for file in files:
        if file.endswith("_1.orun.summary.bench"):
            f = root.split("/sequential")
            bench_files.append((os.path.join(root,file)))

benches = StupidIdea()
benches.add_files(bench_files)

# st.write(benches.structure)
# st.write(bench_files)

n = int(st.text_input('Number of benchmarks','2'))

containers = [st.columns(3) for i in range(n)]

# [[a11, a12 ... a1n], [a21, a22 ... a2n], ... [am1, am2 ... amn]] => [a11]
def flatten(lst):
    return [i for i in chain(*lst)]

# [(a1, b1), (a2, b2) ... (an, bn)] => ([a1, a2, ... an], [b1, b2, ... bn])
def unzip(lst):
    return (list(zip(*lst)))

def unzip_dict(d):
    a = unzip(list(d))
    (x, y) = a[0], flatten(a[1])
    return (x, y)

def fmt_variants(commit, variant):
    return (variant.split('_')[0] + '+' + str(commit) + '_' + variant.split('_')[1])

def unfmt(variant):
    commit = variant.split('_')[0].split('+')[-1]
    variant = variant.split('_')[0].split('+')[0] + '+' + variant.split('_')[0].split('+')[1] + '_' + variant.split('_')[1]
    return (commit , variant)

def get_selected_values(n):
    lst = []
    for i in range(n):
        # create the selectbox in columns
        host_val = containers[i][0].selectbox('hostname', benches.structure.keys(), key = str(i) + '0')
        timestamp_val = containers[i][1].selectbox('timestamp', benches.structure[host_val].keys(), key = str(i) + '1')
        commits, variants = unzip_dict((benches.structure[host_val][timestamp_val]).items())
        # st.write(variants)
        fmtted_variants = [fmt_variants(c, v) for c,v in zip(commits, variants)]
        # st.write(fmtted_variant)
        variant_val = containers[i][2].selectbox('variant', fmtted_variants, key = str(i) + '2')
        selected_commit, selected_variant = unfmt(variant_val)
        lst.append({"host" : host_val, "timestamp" : timestamp_val, "commit" : selected_commit, "variant" : selected_variant})
    return lst

selected_files = StupidIdea()
_ = [selected_files.add(f["host"], f["timestamp"], f["commit"], f["variant"]) for f in get_selected_values(n)]

# Expander for showing bench files
with st.expander("Show metadata for selected benchmarks"):
    st.write(selected_files.structure)

# selected_files = 

# def get_filepath(file):
#     host_val, timestamp_val, variant_val = file[0], file[1], file[2]

#     # create file name
#     commit = variant_val.split('_')[0].split('+')[-1]
#     variant_stem = variant_val.split('_')[1]
#     variant_value = '+'.join(variant_val.split('_')[0].split('+')[:-1]) + '_' + variant_stem

#     file = os.path.join(
#         artifacts_dir,
#         host_val,
#         timestamp_val,
#         commit,
#         variant_value
#     )

#     return file

# def get_dataframe_from_file(file):
#     # json to dataframe
#     data_frames = []

#     with open(file) as f:
#         data = []
#         for l in f:
#             data.append(json.loads(l))
#         df = pdjson.json_normalize(data)
#         value     = file.split('/sequential/')[1]
#         date      = value.split('/')[1].split('_')[0]
#         commit_id = value.split('/')[2][:7]
#         variant   = value.split('/')[3].split('_')[0]
#         df["variant"] = variant + '_' + date + '_' + commit_id
#         data_frames.append(df)

#     df = pd.concat (data_frames, sort=False)
#     df = df.sort_values(['name'])
#     return df

# selected_file_path = [get_filepath(list(file.values())) for file in selected_file_list]
# df_list = [get_dataframe_from_file(file) for file in selected_file_path]

# with st.expander("Show table for the selected benchmarks"):
#     for i in df_list:
#         st.write(i)

# st.header("Select baseline")
# base_container = st.columns(3)
# baseline_h, baseline_t, baseline_v = (
#     base_container[0].selectbox(
#         'hostname', 
#         [file["host"] for file in selected_file_list],
#         key = str(uuid.uuid4())    
#     ),
#     base_container[1].selectbox(
#         'timestamp', 
#         [file["timestamp"] for file in selected_file_list],
#         key = str(uuid.uuid4())    
#     ),
#     base_container[2].selectbox(
#         'variant', 
#         [file["variant"] for file in selected_file_list],
#         key = str(uuid.uuid4())    
#     ))