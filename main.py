#!/usr/bin/env python
# encoding: utf-8

"""
Calculate Spearman correlation coefficient for ranking of n CDVA descriptors
"""
__author__ = "HF"

import sys
import json
import csv

import os
import random
import shutil

import pickle

import math
import time
import numpy as np
import math
from scipy.stats import spearmanr


image_root = r'Landmark_Detection\data\keyframes_2'
num_descriptors = 100   # maximum 100
start_pos = 0
shots = 10

num_args = len(sys.argv) - 1
if num_args == 3:
    num_descriptors = int(sys.argv[1])
    start_pos = int(sys.argv[2])
    shots = int(sys.argv[3])
    print(f'Starting with {num_descriptors} descriptors, at pos {start_pos} using {shots} shots')

########### Print Circuits in LateX
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# import numpy as np
# qc = QuantumCircuit(2)
# phi = np.pi/2
# qc.cz(0,1)
# qc.rx(5*phi,0)
# qc.rx(9*phi,1)
# qc.cx(0,1)
# qc.rx(3*phi,0)
# qc.cx(0,1)
# qc.cz(0,1)
# Urot = qc.to_gate()
# Urot.name = "$U_\mathrm{rot}$"
# ctrl_Urot = Urot.control()
# ctrl_Urot.label = "C-Urot"
# circ = QuantumCircuit(2)
# circ.append(Urot,[0,1])
# circ.draw('latex')
# sys.exit()


# from qrisp import QuantumVariable, GateWrapEnvironment, x, y, z
# qv = QuantumVariable(3)
# gwe = GateWrapEnvironment("U_{rot}")
# with gwe:
#     x(qv[0])
#     y(qv[1])
# z(qv[2])
# print(qv.qs.to_latex())
# sys.exit()

# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# # from qiskit.tools.visualization import circuit_drawer  # old, does not work
# from qiskit.visualization.circuit_visualization import circuit_drawer
# q = QuantumRegister(1)
# c = ClassicalRegister(1)
# qc = QuantumCircuit(q, c)
# qc.h(q)
# qc.measure(q, c)
# print(circuit_drawer(qc, output='latex_source'))
# sys.exit()

# from qrisp import QuantumCircuit
# qc = QuantumCircuit(4, name = "fan out")
# qc.cx(0, range(1,4))
# qc.measure(range(0,4))
# print(qc )
# print(qc.to_latex(initial_state=True, latexdrawerstyle=True))
# sys.exit()





# # Sort the list by the second value of each tuple
# my_list = [(1, 3), (5, 2), (4, 2), (3, 2), (8,4), (9,0)]
# my_list.sort(key=lambda x: x[1])
# # Sort tuples with the same second value by their first value ascending
# done = False
# while not done:
#     again = False
#     for i in range(len(my_list) - 1):
#         if (my_list[i][1] == my_list[i + 1][1]) and my_list[i][0] > my_list[i + 1][0] :
#             my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i] # swap
#             again = True
#     if not again:
#         done = True
#
# print(my_list)
# sys.exit()


def get_next_level_subdirs(rootdir):
    return [os.path.join(rootdir, d) for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]

def get_random_items(my_list, n):
    return random.sample(set(my_list), n)

def get_random_jpg(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Filter the list to only include JPG files
    jpg_files = [file for file in files if file.endswith('.jpg')]
    if len(jpg_files) == 0:
        return None

    # Select a random file from the list
    name = random.choice(jpg_files)
    file = os.path.join(directory, name)
    return str(file)

def copy_file(network_drive, local_subdir):
    # Get the path to the file on the network drive
    # file_path = os.path.join(network_drive, 'file.txt')

    # Copy the file to the local subdirectory
    shutil.copy(network_drive, local_subdir)


def get_filename_without_ext(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_filenames_without_ext(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Remove the file extensions from the list
    filenames = [os.path.splitext(file)[0] for file in files]
    return filenames

########## Descriptor Dataset
"""
5742 descriptors
4148 valid descriptor
143 classes
"""

########## Similarity Example
"""
Similar to 0fbc443f00facd4c.jpg
Dist = 195: 04a0f989bfb03e69.jpg
Dist = 222: 00ea8a839a28e6c8.jpg
Dist = 259: 03488d389c600279.jpg
"""

######### Draw Cirquits for LateX
"""

"""
#######################################################
def search_cdva(id, data):
    for row in data:
        if row['id'] == id:
            return row
    raise Exception(f'missing id {id} in data')

def search_id(id, data):
    for row in data:
        if row['id'] == id:
            return True
    return False

# read ids and cdva strings and converted to 64bit unsigneds
def read_json(file_path, allowed_ids,  bits_per_int=2):
    """
    format: {id: key, feature: binary_array, cdva:string, vals: [converted ints]
    :param file_path:
    :param allowed_ids: the list of keys that are not "noise"
    :param bits_per_int: split string into integers with bits_per_int
    :return:
    """
    items = []
    with open(file_path, 'r') as file:
        json_object = json.load(file)
    for item in json_object:
        id = item['id']
        if ('feature' in item) and (id in allowed_ids):
            if search_id(id, items):  # we can have duplicate ids!
                # raise Exception(f'duplicate id {id} in json')
                # print(f'duplicate id {id} in json')
                continue
            result = ''
            for val in item['feature']:
                result = result + str(val)  # merge binary string
            item['cdva'] = result
            parts = [result[i:i+bits_per_int] for i in range(0, len(result), bits_per_int)]
            vals = []
            for part in parts:
                val = int(part, 2)
                vals.append(val)
            # print(vals)
            item['vals'] = vals
            items.append(item)
    return items

# read the segment_id - which corresponds to id in the json file nad the image_dir
def read_csv(file_path, image_root):
    """
    format: image_dir[0];image_file[1];object_id[2];segment_id[3]
    segment_id=key (in json)
    :param file_path:
    :param image_root: add id only if file exists
    :return: the array of eligible images with 'id' and 'name'

    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        i = 1
        for row in reader:
            # Skip the header row if it exists
            if row[0].lower() == 'image_dir':
                continue
            i += 1
            # Skip images in the "noise" directory
            if row[1].lower().__contains__('noise'):
                continue
            file_path = os.path.join(image_root, row[1])
            if os.path.isfile(file_path):
                id = row[3]
                if id in data:
                    raise Exception(f'duplicate id {id}')
                else:
                    data.append(id)  # {'id': row[3], 'name': row[2]})
            else:
                print(f'no jpg {file_path}')
    print(f'{len(data)} keyframes of {i} selected')
    return data


# def get_name(id, data):
#     for row in data:
#         if row['id'] == id:
#             return row['name']
#     raise Exception(f'{id} not found')

##############################################################################################
# select and copy n random keyframes to local keyframes dir
# return only valid keyframes
valid_ids = read_csv(r'cdva\keyframes_v2.csv', image_root=image_root)
print(f'found {len(valid_ids)} csv items')
# images are at data\keyframes_v2\
# don't use images in "noise" subdirs
cdva1 = read_json(r'cdva\keyframes_v2_features_cdva.json', valid_ids,  bits_per_int=64)
print(f'verified {len(cdva1)} json items')

# # select and copy randomly cdva descriptoors from the different classes
# max_count = 99
# subdirs = get_next_level_subdirs(image_root)
# selected_subdirs = get_random_items(subdirs, max_count)
# files = []
# for subdir in selected_subdirs:
#     file = get_random_jpg(subdir)
#     if file:
#         id = get_filename_without_ext(file)
#         if id in valid_ids:
#             files.append(file)  # for copy to local drive
#         else:
#             raise Exception(f'something wrong with id {id}')
#
# for file in files:
#     copy_file(file, r'keyframes')

desc_length = 0
final_cdva = []
ids = get_filenames_without_ext(r'keyframes')

for i, id in enumerate(ids):
    item = search_cdva(id, cdva1)
    final_cdva.append(item)
    if desc_length == 0:
        desc_length = len(item['cdva'])
    # print(i+1, id, item['cdva'][:20], item['vals'])

# sys.exit()
####################################################


# i = 0
# for item in cdva1:
#     i += 1
#     if 'id' in item:
#         key = item['id']
#         val = get_name(key, cdva2)
#         item['name'] = val
#         print(i, key, val)

########################

def hamming_distance(x, y):
    if isinstance(x, int):
        return bin(x ^ y).count('1')
    result = 0
    if isinstance(x, list):
        for i, xval in enumerate(x):
            result += bin(xval ^ y[i]).count('1')
        return result
    raise Exception("Unknown data")


def sort_and_extract_classic(my_list):
    """
    Sort ascending on tuple[2], but put tuple[1] in the result list
    :param my_list: list of tuples, first=index, second=hamming distance
    :return: sorted list of indices with most similar one first
    """
    # return [x[0] for x in sorted(my_list, key=lambda x: x[1])]  # not enough, does not consider equal similarity
    my_list.sort(key=lambda x: x[1])
    # Sort tuples with the same second value by their first value ascending
    done = False
    while not done:
        again = False
        for i in range(len(my_list) - 1):
            if (my_list[i][1] == my_list[i + 1][1]) and my_list[i][0] > my_list[i + 1][0] :
                my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i] # swap
                again = True
        if not again:
            done = True
    return my_list

# print(binary_to_unsigned_int('1010'))  # 10
# print(binary_to_unsigned_int('11111111111111111111111111111111'))  # 4294967295 = 2 ^ 32 - 1
# print(binary_to_unsigned_int('1111111111111111111111111111111111111111111111111111111111111111'))  # 18446744073709551615 = 2 ^ 64 - 1

# print(hamming_distance(10,8))  # 1010 ^ 1000 = 0010 => 1 bit diff
# print(hamming_distance([10],[9]))  # 1010 ^ 1001 = 0011 => 2 bits diff
#
# l_4 = [(0,2),(1,3), (2,1), (3,0)]
# print(sort_and_extract_classic(l_4))  # [3,2,0,1] => index 3 is the most similar entry to index 4

cdva = final_cdva[start_pos:start_pos+num_descriptors]
assert len(cdva) == num_descriptors, f'Invalid cdva length {len(cdva)}'
print(f'Check CDVAs from pos {start_pos} to {start_pos + num_descriptors}, ({num_descriptors} items)')
#####################################################################
# # # test with 3 dummy items to calculate the all-to-all rankings - must be multiple of 2^n
# cdva=[
#     {'id': 0, 'cdva': '01010101', 'vals': [1, 1, 1, 1]},  # simulate sizes to figure out the quantum gate depth
#     {'id': 1, 'cdva': '01011001', 'vals': [1, 1, 2, 1]},
#     {'id': 2, 'cdva': '01011101', 'vals': [1, 1, 3, 1]},
# ]
"""
(desc_len=8, 3 qubits, cirquit max_depth 22, min_depth 16, avg_depth 19.0, qrisp: 26, 26 => 26.0)
desc_len=64, 7 qubits,  cirquit max_depth 134, min_depth 128, avg_depth 131.0, qrisp: 250, 250 => 250.0
desc_len=128, 8 qubits,  cirquit max_depth 262, min_depth 256, avg_depth 259.0, qrisp: 506, 506 => 506.0
desc_len=512, 10 qubits,  cirquit max_depth 1030, min_depth 1024, avg_depth 1027.0, qrisp: 2042, 2042 => 2042.0
desc_len=1024, 11 qubits,  cirquit max_depth 2054, min_depth 2048, avg_depth 2051.0, qrisp: 4090, 4090 => 4090.0
desc_len=2048, 12 qubits,  cirquit max_depth 4102, min_depth 4096, avg_depth 4099.0, qrisp: 8186, 8186 => 8186.0
desc_len=4096, 13 qubits,  cirquit max_depth 8198, min_depth 8192, avg_depth 8195.0, qrisp: 16378, 16378 => 16378.0
desc_len=8192, 14 qubits,  cirquit max_depth 16390, min_depth 16384, avg_depth 16387.0, qrisp: 32762, 32762 => 32762.0
desc_len=16384, 15 qubits,  cirquit max_depth 32774, min_depth 32768, avg_depth 32771.0, qrisp: 65530, 65530 => 65530.0
"""
####################################################################
n = math.ceil(math.log2(len(cdva[0]['cdva']))) + 1  # number of qubits


print('CPU check...')
ranking_a = []
for i, item1 in enumerate(cdva):
    val1 = item1['vals']  # list of ints
    result = []
    ranking_a.append([])
    for j, item2 in enumerate(cdva):
        if i == j:
            continue
        # print('checking', i , 'against', j)
        val2 = item2['vals']
        dist = hamming_distance(val1, val2)
        result.append((j, dist))  # the lower, the more similar
    ranking_a[i].append(sort_and_extract_classic(result))
    assert len(ranking_a) - 1 == i

print(f'ranking index ranking_a[i] = ranking for target i, compared to the others')
print(f'[[[2, 1]], ...] means, item 0 is more similar to 2 than to 1')
print(ranking_a)

# # for test: show images
# print(f'Similar to {cdva[0]["id"]}.jpg')
# for i in range(3):
#     print(f'Dist = {ranking_a[0][0][i][1]}: {cdva[ranking_a[0][0][i][0]]["id"]}.jpg')
# sys.exit()

###################################################################
# create ranking_b on quantum side
import numpy as np
import math
from importlib.metadata import version
# print(version("qrisp"))  # 0.4.9
# print(version("qiskit"))  # 1.1.2

import qiskit_aer as Aer
qiskit_backend = Aer.AerSimulator()  # get_backend('qasm_simulator')
from qiskit import transpile as qiskit_transpile
execute = qiskit_backend.run
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation
# for qrisp depth
from qrisp import QuantumCircuit as qrisp_QuantumCircuit

def sort_and_extract_quantum(my_list):
    """
    Sort ascending on tuple[2], but put tuple[1] in the result list
    Items with the same tuple[1] similarity, are sorted ascending by tuple[0]
    :param my_list: list of tuples, first=index, second=hamming distance
    :return: sorted list of indices with most similar one first
    """
    # return [x[0] for x in sorted(my_list, reverse=True, key=lambda x: x[1])]  # does not consider equal similarity
    my_list.sort(reverse=True, key=lambda x: x[1])
    # Sort tuples with the same second value by their first value ascending
    done = False
    while not done:
        again = False
        for i in range(len(my_list) - 1):
            if (my_list[i][1] == my_list[i + 1][1]) and my_list[i][0] > my_list[i + 1][0] :
                my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i] # swap
                again = True
        if not again:
            done = True
    return my_list


def encode_bitstring(bitstring, qr, cr, inverse=False):
    """
    create a circuit for constructing the quantum superposition of the bitstring
    """
    n = math.ceil(math.log2(len(bitstring))) + 1  # number of positional qubits + value qubit
    assert n > 2, "the length of bitstring must be at least 2"

    qc = QuantumCircuit(qr, cr)

    # the probability amplitude of the desired state
    desired_vector = np.array([0.0 for i in range(2 ** n)])  # initialize to zero
    amplitude = np.sqrt(1.0 / 2 ** (n - 1))

    for i, b in enumerate(bitstring):
        pos = i * 2  # for |0>
        if b == "1":
            pos += 1  # for |1>
        desired_vector[pos] = amplitude
    if not inverse:
        qc.initialize(Statevector(desired_vector), qr)  # With Statevector
        qc.barrier(qr)
    else:
        # see https://stackoverflow.com/questions/77971628/circuiterror-inverse-not-implemented-for-reset
        state_prep = StatePreparation(Statevector(desired_vector))
        qc.append(state_prep, qr)
        # qc.initialize(desired_vector,qr)  # does not work: 'inverse-not-implemented-for-reset'

        # qc = transpile(qc, backend).inverse()  # does not work in 1.1.1
        qc = qiskit_transpile(qc, qiskit_backend).inverse()

        for i in range(n):
            qc.measure(qr[i], cr[i])
    return qc

print('QPU check...')
print(f'n={n} qubits, shots={shots}')
qr = QuantumRegister(n)
cr = ClassicalRegister(n)

# Create circuit templates
max_circuit_depth = 0
min_circuit_depth = sys.maxsize
avg_circuit_depth = 0
# qrisp
qr_maxd = 0
qr_mind = sys.maxsize
qr_avgd = 0

circs = []
inverse_circs = []
start = time.time()
for i, item in enumerate(cdva):
    print(f'prepare {i} of {len(cdva)}')
    circs.append(encode_bitstring(item['cdva'], qr, cr))
    inverse_circs.append(encode_bitstring(item['cdva'], qr, cr, inverse=True))
end = time.time()
prep_time = end-start
print(f'preparation time = {prep_time} s')

def quantum_distance(n, i, circs, j, inverse_circs, shots):
    global max_circuit_depth, min_circuit_depth, qr_mind, qr_maxd
    combined_circ = circs[i].compose(inverse_circs[j])
    new_circuit = qiskit_transpile(combined_circ, qiskit_backend)
    # gate depth calculation
    # d = new_circuit.depth()
    # if d > max_circuit_depth:
    #     max_circuit_depth = d
    # if d < min_circuit_depth:
    #     min_circuit_depth = d
    # qrisp_circuit = qrisp_QuantumCircuit.from_qiskit(new_circuit)
    # qct = qrisp_circuit.transpile()
    # d = qct.cnot_depth()
    # if d > qr_maxd:
    #     qr_maxd = d
    # if d < qr_mind:
    #     qr_mind = d
    job = execute(new_circuit, backend=qiskit_backend, shots=shots)
    st = job.result().get_counts(combined_circ)
    if "0" * n in st:
        sim_score = int(st["0" * n])
    else:
        # raise Exception(f'no simulation score for {(i, j)}')
        sim_score = 0  # no similarity
    return sim_score

start = time.time()
# calc QPU similarity
ranking_b = []
for i, item1 in enumerate(cdva):
    print(i,'...')
    result = []
    ranking_b.append([])
    for j, item2 in enumerate(cdva):
        if i == j:
            continue
        # print('checking', i , 'against', j)
        dist = quantum_distance(n, i, circs, j, inverse_circs, shots)
        result.append((j, dist))  # different distance, the higher, the more similar
    ranking_b[i].append(sort_and_extract_quantum(result))
    assert len(ranking_b) - 1 == i

print(f'ranking index ranking_a[i] = ranking for target i, compared to the others')
# print(f'[[[2, 1]], ...] means, item 0 is more similar to 2 than to 1')

end = time.time()

print(ranking_a)
print(ranking_b)
print('--')
print([ranking_a[i][0] for i in range(len(ranking_a))])
print([ranking_b[i][0] for i in range(len(ranking_b))])
#print(f'a vs. b = {ranking_a == ranking_b}')

sim_time = end-start
print(f"Time taken: {sim_time} seconds")

# save rankings
with open(f'ranking_a_{start_pos}_{num_descriptors}_{shots}.pkl', 'wb') as f:
    pickle.dump(ranking_a, f)
with open(f'ranking_b_{start_pos}_{num_descriptors}_{shots}.pkl', 'wb') as f:
    pickle.dump(ranking_b, f)



###################################################################
# # calculate Spearman's correlation coefficient and p-value
# # spearman rank correlation test
# from scipy.stats import spearmanr
#
# # sample data
# x = [1, 2, 3, 4, 5]
# y = [5, 4, 3, 2, 1]
#
# # calculate Spearman's correlation coefficient and p-value
# corr, pval = spearmanr(x, y)
#
# # print the result
# print("Spearman's correlation coefficient:", corr)  # should be close to -1 (-0.9999999999999999)
# print("p-value:", pval)  # 1.4042654220543672e-24

corr_sum = 0.0
min_score = 2
max_score = -2
from scipy.stats import spearmanr

for i, item in enumerate(ranking_a):
    # extract index into lists
    l = item[0]  # [0] is index in cdva list, [1] is
    # for monotony check assign ground truth ascending keys
    mapping = dict()
    for j, val in enumerate(l):
        mapping[val[0]] = j  # mapping[index in cdva] = ascending order
    x = [k for k in range(len(l))]
    l = ranking_b[i][0]  # [0] = index, must be matched to ranking_a
    y = []
    for k, val in enumerate(l):
        y.append(mapping[val[0]])  # mapping [index in cdva] = mapping[ranking_a]
    print(l)
    # y = [val[1] for val in l]
    # print(i)
    print(x)
    print(y)
    corr, pval = spearmanr(x, y)
    print('corr=', corr)
    if corr < min_score:
        min_score = corr
    if corr > max_score:
        max_score = corr
    corr_sum += corr

l = len(ranking_a)
corr_avg = corr_sum / l
print(f'Avg score over {l} rankings at {shots} shots=', corr_avg)
avg_circuit_depth = (max_circuit_depth + min_circuit_depth)/2.0
qr_avgd = (qr_maxd + qr_mind) / 2.0
assert 2**(n-1) == desc_length, "What?!"
d = f'desc_len={2**(n-1)}, {n} qubits,  cirquit max_depth {max_circuit_depth}, min_depth {min_circuit_depth}, avg_depth {avg_circuit_depth}, qrisp: {qr_maxd}, {qr_mind} => {qr_avgd}'
print(d)
with open(f'results_{start_pos}_{num_descriptors}_{shots}.txt', 'w') as f:
    print(f'Score for {num_descriptors} desc with {desc_length} bits, starting from pos {start_pos}, using {shots} shots, avg_score={corr_avg}; min_score={min_score}; max_score={max_score}; prep time[s] ;{prep_time}; sim_time[s] ;{sim_time}', file=f)
    for i, item_a in enumerate(ranking_a):
        item_b = ranking_b[i]
        print(i)
        print(item_a)
        print(item_b)

