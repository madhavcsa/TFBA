# -*- coding: utf-8 -*-
#!usr/bin/env python

import sys
import numpy as np
import argparse
import os
import operator
import pickle

def mergeSchemas(schemas):
	
	h_schemas = []
	if len(schemas) == 1:
		return h_schemas

	for i1 in range(len(schemas)):
		for i2 in range(i1+1, len(schemas)):
			schm1 = schemas[i1]
			schm2 = schemas[i2]
			if schm1 != schm2:
					if schm1[:2] == schm2[:2]:
						h_schemas.append([schm1[0],schm1[1],schm1[2],schm2[2]])

	h_schemas = [tuple(a) for a in h_schemas]
	h_schemas = set(h_schemas)
	h_schemas = list(h_schemas)
	h_schemas = [list(a) for a in h_schemas]

	return h_schemas




def getSchema(ids1, ids2, ids3):


	schemas = []
	for idx1 in ids1:
		for idx2 in ids2:
			l = []
			if idx1.split(",")[1] == idx2.split(",")[0]:
				l.append(int(idx1.split(",")[0].strip()))
				for idx3 in ids3:
					if idx3.split(",")[1] == idx2.split(",")[1] and idx1.split(",")[0] == idx3.split(",")[0]:
						l.append(int(idx2.split(",")[0].strip()))
						l.append(int(idx2.split(",")[1].strip()))
						schemas.append(l)

		
	return schemas


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data", help="Enter the path to data dir", type=str)
	parser.add_argument("odir", help="Enter the path to Output Folder", type=str)
	parser.add_argument("--rank", nargs = '+', type = int)
	args = parser.parse_args()

	dir = args.data

	odir = args.odir
	
	rank = args.rank
	print "Loading Data"	
	agents = pickle.load(open(os.path.join(dir,"agents"),"rb"))
	patnts = pickle.load(open(os.path.join(dir,"patients"),"rb"))
	instmnts = pickle.load(open(os.path.join(dir,"instruments"),"rb"))
	predicates = pickle.load(open(os.path.join(dir, "predicates"), "rb"))

	G = pickle.load(open(os.path.join(odir, "Core"),"rb"))
	Factors = pickle.load(open(os.path.join(odir, "Factors"), "rb"))


	schmea_dict = {}

	## Writing Schemas
	dictIDS1 = {}
	dictIDS2 = {}
	dictIDS3 = {}
	idx = 0
	for i in range(rank[0]):
		for j in range(rank[1]):			
			dictIDS1[idx] = str(i)+","+str(j)
			idx += 1

	idx = 0
	for i in range(rank[1]):
		for j in range(rank[2]):			
			dictIDS2[idx] = str(i)+","+str(j)
			idx += 1

	idx = 0
	for i in range(rank[0]):
		for j in range(rank[2]):			
			dictIDS3[idx] = str(i)+","+str(j)
			idx += 1


	dictIDS={}
	dictIDS[0] = dictIDS1
	dictIDS[1] = dictIDS2
	dictIDS[2] = dictIDS3

	tops = 6
	Agents =[]
	Patients = []
	Instruments = []

	for col in range(rank[0]):
		topIds = np.argsort(Factors[0][:,col])[::-1][:tops]
		l = []
		for id in topIds:			
			l.append(agents[id])

		Agents.append(l)


	for col in range(rank[1]):
		topIds = np.argsort(Factors[1][:,col])[::-1][:tops]
		l = []
		for id in topIds:			
			l.append(patnts[id])

		Patients.append(l)



	for col in range(rank[2]):
		topIds = np.argsort(Factors[2][:,col])[::-1][:tops]
		l = []
		for id in topIds:
			l.append(instmnts[id])

		Instruments.append(l)



	for p in range(len(predicates)):		
		ids1 = np.argsort(G[0][:,:,p], axis=None)[::-1][:8]
		ids2 = np.argsort(G[1][:,:,p], axis=None)[::-1][:8]
		ids3 = np.argsort(G[2][:,:,p], axis=None)[::-1][:8]

		idx1 = [dictIDS1[idx] for idx in ids1]
		idx2 = [dictIDS2[idx] for idx in ids2]
		idx3 = [dictIDS3[idx] for idx in ids3]

		schemas = getSchema(idx1[:4],idx2[:4],idx3[:4])

		#if len(schemas) == 0:
		#	schemas = getSchema(idx1,idx2,idx3)

		h_schemas = mergeSchemas(schemas)
		if len(h_schemas)>0:
			for s in range(len(h_schemas)):
				schm = h_schemas[s]

				score = 2*G[0][schm[0],schm[1],p] + G[1][schm[1],schm[2],p] + G[2][schm[0],schm[2],p] + G[2][schm[0],schm[3],p] + G[1][schm[1],schm[3],p]
				score = score/6

				schm.append(p)
				schmea_dict[tuple(schm)] = score
				

		if len(schemas)>0:
			for s  in range(len(schemas)):
				schm = schemas[s]
				
				score = G[0][schm[0],schm[1],p] + G[1][schm[1],schm[2],p] + G[2][schm[0],schm[2],p]
				score = score/3
				
				schm.append(p)
				schmea_dict[tuple(schm)] = score

	sorted_key = sorted(schmea_dict.items(), key=operator.itemgetter(1))[::-1]
	sp = open(os.path.join(odir, "TopSchemas.txt"),'w')
	for key in sorted_key[:50]:

		sp.write(predicates[int(key[0][len(key[0])-1])]+":\n")
		
		sp.write(str(Agents[int(key[0][0])])+"\n")
		sp.write(str(Patients[int(key[0][1])])+"\n")
		sp.write(str(Instruments[int(key[0][2])])+"\n")
		if len(key[0]) == 5:
			sp.write(str(Instruments[int(key[0][3])])+"\n")
		sp.write("\n")


	sp.close()



	








	