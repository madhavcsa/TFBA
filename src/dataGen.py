# -*- coding: utf-8 -*-
#!usr/bin/env python

import sys
import numpy as np
import argparse
import os
import math
import pickle
import datetime
from sktensor import sptensor
from sktensor import dtensor




if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument("input", help = "Enter the path to input file", type=str)
	parser.add_argument("odir", help="Enter the path to output directory", type=str)


	args = parser.parse_args()

	agents = []
	predicates = []
	patients = []
	instruments = []

	dictTrp1 = {}
	dictTrp2 = {}
	dictTrp3 = {}
	
	for line in open(args.input).readlines():
		tok = line.split('\t')
		if tok[0] not in agents:
			agents.append(tok[0])
		if tok[1] not in predicates:
			predicates.append(tok[1])
		if tok[2] not in patients:
			patients.append(tok[2])
		if tok[3] not in instruments:
			instruments.append(tok[3])		

		tup1 = (agents.index(tok[0]), patients.index(tok[2]), int(tok[4]))

		if tok[3] != "<na>":

			tup2 = (patients.index(tok[2]), instruments.index(tok[3]), int(tok[4]))
			tup3 = (agents.index(tok[0]), instruments.index(tok[3]), int(tok[4]))

		p = predicates.index(tok[1])
		if p in dictTrp1:

			flag = 0			
			for i in range(len(dictTrp1[p])):
				l_i = dictTrp1[p][i]				
				if [l_i[0],l_i[1]] == [tup1[0],tup1[1]]:
					l_i = tuple([l_i[0],l_i[1],l_i[2]+tup1[2]])
					flag = 1
					dictTrp1[p][i] = l_i
					break

			if flag == 0:
				dictTrp1[p].append(tup1)	

			if tok[3] != "<na>":
				dictTrp2[p].append(tup2)
				dictTrp3[p].append(tup3)


			
			
		else:
			dictTrp1[p] = []
			dictTrp2[p] = []
			dictTrp3[p] = []
			dictTrp1[p].append(tup1)
			if tok[3] != "<na>":
				dictTrp2[p].append(tup2)
				dictTrp3[p].append(tup3)


	I = len(agents)
	J = len(patients)
	K = len(instruments)

	points_a = []
	points_b = []
	points_c = []
	vals = []

	for k in range(len(predicates)):
		tups = dictTrp1[k]
		for tup in tups:
			points_a.append(tup[0])
			points_b.append(tup[1])
			points_c.append(k)
			vals.append(tup[2])

	L = []
	L.append(points_a)
	L.append(points_b)
	L.append(points_c)
	X1 = sptensor(tuple(L), vals, shape =(I,J,len(predicates)), dtype=float)

	points_a = []
	points_b = []
	points_c = []
	vals = []

	for k in range(len(predicates)):
		tups = dictTrp2[k]
		for tup in tups:			
			points_a.append(tup[0])
			points_b.append(tup[1])
			points_c.append(k)
			vals.append(tup[2])

	L = []
	L.append(points_a)
	L.append(points_b)
	L.append(points_c)
	X2 = sptensor(tuple(L), vals, shape =(J, K, len(predicates)), dtype=float)

	points_a = []
	points_b = []
	points_c = []
	vals = []

	for k in range(len(predicates)):
		tups = dictTrp3[k]
		for tup in tups:
			points_a.append(tup[0])
			points_b.append(tup[1])
			points_c.append(k)
			vals.append(tup[2])

	L = []
	L.append(points_a)
	L.append(points_b)
	L.append(points_c)
	X3 = sptensor(tuple(L), vals, shape =(I, K, len(predicates)), dtype=float)


	dataFolder = args.odir
	if not os.path.exists(dataFolder):
		os.makedirs(dataFolder)
	pickle.dump(agents, open(os.path.join(dataFolder,"agents"), "wb"))
	pickle.dump(predicates, open(os.path.join(dataFolder,"predicates"),"wb"))
	pickle.dump(patients, open(os.path.join(dataFolder,"patients"),"wb"))
	pickle.dump(instruments, open(os.path.join(dataFolder,"instruments"),"wb"))
	pickle.dump(X1, open(os.path.join(dataFolder, "X1"),"wb"))
	pickle.dump(X2, open(os.path.join(dataFolder, "X2"),"wb"))
	pickle.dump(X3, open(os.path.join(dataFolder, "X3"),"wb"))
	



