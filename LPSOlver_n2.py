import numpy as np
from scipy.optimize import linprog
from numpy import genfromtxt
import random
import sys
from datetime import datetime
import os

"""
folder = 'data/'
filename = 'DOTDec2017'
columns = [10,11,16,13,19,20,12,14]
"""

"""
scipy.optimize.linprog(
	c,
	A_ub=None,
	b_ub=None,
	A_eq=None,
	b_eq=None,
	bounds=None,
	method='simplex',
	callback=None,
	options=None
)

Minimize:     c^T * x

Subject to:   A_ub * x <= b_ub
              A_eq * x == b_eq
"""

Optimizer = "simplex" #"interior-point"
#K = 5
if len(sys.argv) < 5:
	print("Please input sufficient arguments")
	sys.exit()
N = int(sys.argv[1])
K = int(sys.argv[2])
red_point_ratio = int(sys.argv[3])
output_filename = sys.argv[4]

"""
Regret ratiowhen {t1, t2} is replaced by t1
t1: First tuple
t2: Second tuple
region: Constraint region - [Region, np.zeros(N,1)]
"""
def _get_RR_btwn_tpls(t1, t2, region):
	if region is  None:
		A_ineq = np.hstack((t2 - t1, np.ones(1))).reshape(1, -1)
	else:
		A_ineq = np.vstack((region, np.hstack((t2 - t1, np.ones(1)))))
	A_eq = np.hstack((t1, np.zeros(1))).reshape(1, -1)
	b_eq = np.ones(1)
	b_ineq = np.zeros((1, A_ineq.shape[0]))
	c = np.zeros((A_ineq.shape[1]))
	c[-1] = -1
	return linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,method=Optimizer)

def _form_region(tpl, other_tuples):
	return np.hstack((other_tuples - tpl, np.zeros((other_tuples.shape[0], 1))))

def _get_RR_for_tpl(database, index, worst_score, region):
	max_RR = 0
	t1 = database[index, :]
	for i in range(database.shape[0]):
		#print("Iteration " + str(i))
		if i == index:
			continue
		t2 = database[i, :]
		soln = _get_RR_btwn_tpls(t1, t2, region)
		if not soln.success:
			if soln.message == "The algorithm terminated successfully and determined that the problem is infeasible.":
				return 1
			elif soln.message == "Optimization failed. Unable to find a feasible starting point.":
				return 1
			else:
				return 1
		elif soln.x[-1] > 1.01:
			_get_RR_btwn_tpls(t1, t2, region)
		max_RR = max(max_RR, soln.x[-1])
		if max_RR > worst_score:
			return max_RR
	return max_RR

def _get_RR_for_set(database, kmedoid_indices):
	RR = []
	set_medoid_indices = set(kmedoid_indices)
	for i in kmedoid_indices:
		other_indices = list(set_medoid_indices - set([i]))
		other_tuples = database[other_indices, :]
		RR.append(_get_RR_for_tpl(database, i, 1, _form_region(database[i, :], other_tuples)))
	return RR

def get_new_medoids(database, kmedoid):
	k = kmedoid.shape[0]
	medoid_indices = []
	rr = []
	regret_ratio = 0
	for index in range(kmedoid.shape[0]):
		tpl = kmedoid[index, :]
		other_indices = list(set(range(k)) - set([index]))
		other_tuples = kmedoid[other_indices, :]
		region = _form_region(tpl, other_tuples)
		worst_score = 1
		cIndex = -1
		for i in range(database.shape[0]):
			#print(str(index) + " " + str(i) + " Iteration") 
			w_score = _get_RR_for_tpl(database, i, worst_score, region)
			if w_score < worst_score:
				worst_score = w_score
				cIndex = i
		regret_ratio = max(regret_ratio, worst_score)
		rr.append(regret_ratio)
		medoid_indices.append(cIndex)
	#print(rr)
	return medoid_indices

def get_new_fair_medoids(database, kmedoid, type_column, kmedoid_indices):
	k = kmedoid.shape[0]
	medoid_indices = []
	rr = []
	regret_ratio = 0
	for index in range(kmedoid.shape[0]):
		tpl = kmedoid[index, :]
		other_indices = list(set(range(k)) - set([index]))
		other_tuples = kmedoid[other_indices, :]
		region = _form_region(tpl, other_tuples)
		worst_score = 1
		cIndex = -1
		for i in range(database.shape[0]):
			if type_column[i]==type_column[kmedoid_indices[index]]:
				continue
			#print(str(index) + " " + str(i) + " Iteration") 
			w_score = _get_RR_for_tpl(database, i, worst_score, region)
			if w_score < worst_score:
				worst_score = w_score
				cIndex = i
		regret_ratio = max(regret_ratio, worst_score)
		rr.append(regret_ratio)
		medoid_indices.append(cIndex)
	#print(rr)
	return medoid_indices


def _check_feasibility(medoids, tpl):
	if len(medoids) == 0:
		return True
	A_ineq = medoids - tpl
	A_eq = tpl.reshape(1, -1)
	b_ineq = np.zeros((A_ineq.shape[0], 1))
	b_eq = np.ones((1, 1))
	c = np.zeros(A_ineq.shape[1])
	soln = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,method=Optimizer)
	return soln.success

def _verify_get_RR_old_region(database, old_indices, new_indices):
	RR = []
	set_old_indices = set(old_indices)
	for i in range(len(old_indices)):
		other_tuples = database[list(set_old_indices - set([old_indices[i]])), :]
		old = database[old_indices[i], :]
		RR.append(_get_RR_for_tpl(database, new_indices[i], 1, _form_region(old, other_tuples)))
	return RR
	
def _get_starting_medoids(database, k):
	"""
	medoid_indices = []
	for i in range(k):
		medoids = database[medoid_indices]
		if len(medoid_indices) == 0:
			medoid_indices.append(i)
		elif _check_feasibility(medoids, database[i, :]):
			medoid_indices.append(i)
	return medoid_indices
	"""
	"""
	medoid_indices = []
	# Generate serial medoid
	i = 0
	index = 0
	while i < k:
		if _get_RR_for_tpl(database, index, 1, None) < 0.85 and _check_feasibility(database[medoid_indices], database[index, :]):
			medoid_indices.append(index)
			i += 1
		index += 1
	return medoid_indices
	"""
	medoid_indices = []
	# Generate random medoid
	i = 0
	index = 0
	while i < k:
		index = random.randint(0,database.shape[0]-1)
		if index in medoid_indices:
			continue
		#if _check_feasibility(database[medoid_indices], database[index, :]):
		if _get_RR_for_tpl(database, index, 1, None) < 0.99 and _check_feasibility(database[medoid_indices], database[index, :]):
			medoid_indices.append(index)
			i += 1
		index += 1
	return medoid_indices

def _get_fair_starting_medoids(database, k, type_column, red_point_ratio):
	"""
	medoid_indices = []
	for i in range(k):
		medoids = database[medoid_indices]
		if len(medoid_indices) == 0:
			medoid_indices.append(i)
		elif _check_feasibility(medoids, database[i, :]):
			medoid_indices.append(i)
	return medoid_indices
	"""
	"""
	medoid_indices = []
	# Generate serial medoid
	i = 0
	index = 0
	while i < k:
		if _get_RR_for_tpl(database, index, 1, None) < 0.85 and _check_feasibility(database[medoid_indices], database[index, :]):
			medoid_indices.append(index)
			i += 1
		index += 1
	return medoid_indices
	"""
	medoid_indices = []
	# Generate random medoid
	i = 0
	index = 0
	# number of red points required
	num_reds = int((red_point_ratio/100)*len(k))
	num_blues = k - num_reds
	while i < k:
		index = random.randint(0,database.shape[0]-1)
		if index in medoid_indices or (num_reds==0 and type_column[index]==0) or (num_blues==0 and type_column[index]==1):
			continue
		#if _check_feasibility(database[medoid_indices], database[index, :]):
		if _get_RR_for_tpl(database, index, 1, None) < 0.99 and _check_feasibility(database[medoid_indices], database[index, :]):
			medoid_indices.append(index)
			if type_column[index]==0: num_reds-=1
			else: num_blues-=1
			i += 1
		index += 1
	return medoid_indices

def generate_type_column(num_rows, red_point_ratio):
	try:
		num_reds = int((red_point_ratio/100)*num_rows)
		type_column_values = [0]*num_reds + [1]*(num_rows-num_reds)
		random.shuffle(type_column_values)
		return type_column_values
	except Exception as e:
		print(f"Exception occurred during type column generation: {e}")
		sys.exit()
	
if __name__ == "__main__":
	file_path = os.path.join('.', 'data','DOTDec2017.csv')
	my_data = genfromtxt(file_path, delimiter=',')
	columns = [10,11,16,13,19,20,12,14]
	
	# generate a column to divide the database points into two classes
	type_column = generate_type_column(len(my_data), red_point_ratio)

	# import pdb
	# pdb.set_trace()
	my_data = my_data[1:N+1, columns]
	database = my_data
	kmedoid_indices = _get_starting_medoids(database, K)
	kmedoids = database[kmedoid_indices, :]
	print(kmedoid_indices)
	print(_get_RR_for_set(database, kmedoid_indices))
	for iter in range(1):
		old_kmedoid_indices = kmedoid_indices
		start_time = datetime.now()
		kmedoid_indices = get_new_medoids(database, kmedoids)
		end_time = datetime.now()
		with open(output_filename, "w") as file:
			file.write(str(N) + " " + str(K) + "\n")
			file.write(str((end_time - start_time).total_seconds()))
		kmedoids = database[kmedoid_indices, :]
		print(_verify_get_RR_old_region(database, old_kmedoid_indices, kmedoid_indices))
		print("Iteration " + str(iter + 1))
		print(kmedoid_indices)
		print(_get_RR_for_set(database, kmedoid_indices))