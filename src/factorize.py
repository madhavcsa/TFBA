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
from sktensor import tucker
from sktensor.core import norm
from scipy.sparse import rand as sprand
from joblib import Parallel, delayed
import multiprocessing
from numpy import sqrt
from joblib import Parallel, delayed


global X1
global X2 
global X3
global agents
global patnts
global instmnts
global predicates


def getVal(S, idx):
	A = np.array(S.subs)
	for i in range(len(S.vals)):
		if list(A[:,i]) == idx:
			return S.vals[i]

	return 0

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




def loadData(dataDir):


	global X1
	global X2 
	global X3
	global agents
	global patnts
	global instmnts
	global predicates

	#try:
	print "Loading Data"
	print dataDir
	X1 = pickle.load(open(os.path.join(dataDir,"X1"),"rb"))
	X2 = pickle.load(open(os.path.join(dataDir,"X2"),"rb"))
	X3 = pickle.load(open(os.path.join(dataDir,"X3"),"rb"))
	agents = pickle.load(open(os.path.join(dataDir,"agents"),"rb"))
	patnts = pickle.load(open(os.path.join(dataDir,"patients"),"rb"))
	instmnts = pickle.load(open(os.path.join(dataDir,"instruments"),"rb"))
	predicates = pickle.load(open(os.path.join(dataDir, "predicates"), "rb"))

#		return X1, X2, X3, agents, patnts, instmnts, predicates



	#except:
	#	print "Error Loading Data"


def Initialize(rank):
	

	U1,G1 = tucker.hosvd(X1, [rank[0],rank[1], len(predicates)-1])
	U2,G2 = tucker.hosvd(X2, [rank[1],rank[2], len(predicates)-1])
	U3,G3 = tucker.hosvd(X3, [rank[0],rank[2], len(predicates)-1])

	Factors = []
	Factors.append(np.abs(U1[0])+np.abs(U3[0]))
	Factors.append(np.abs(U1[1])+np.abs(U2[0]))
	Factors.append(np.abs(U2[1])+np.abs(U3[1]))

	G = []
	
	G.append(np.random.normal(0,2,(rank[0],rank[1],len(predicates))).clip(0.5))
	G.append(np.random.normal(0,2,(rank[1],rank[2],len(predicates))).clip(0.5))
	G.append(np.random.normal(0,2,(rank[0],rank[2],len(predicates))).clip(0.5))

	return Factors, G



def NN_Factorize(rank, Iters, odir, lambdas, FIT):


	gstart = datetime.datetime.now()

	Factors, G = Initialize(rank)

	print "Initialization Done\n" 

	I = X1.shape[0]
	J = X1.shape[1]
	K = X2.shape[1]

	print "*** Input Summary ***\n"
	print "X1 : "+ str(X1.shape)
	print "X2 : "+ str(X2.shape)
	print "X3 : "+ str(X3.shape)
	print "Factorization rank: "+str(rank)

	lambda_a = lambdas[0]
	lambda_b = lambdas[1]
	lambda_c = lambdas[2]
	
	
	odir = os.path.join(args.odir,"_"+str(rank[0])+"_"+str(rank[1])+"_"+str(rank[2])+"_"+str(lambda_a)+"_"+str(lambda_b)+"_"+str(lambda_c))
	ep = 1e-9
	conv = 1e-9

	Logs = os.path.join(odir,"Logs")
	
	if not os.path.exists(Logs):
		os.makedirs(Logs)

	
	normX1 = norm(X1)
	normX2 = norm(X2)
	normX3 = norm(X3)

	fit = np.zeros(3)
	avgFit = 0

	for iter in range(1,Iters+1):
		print "\n Starting Iteration: "+str(iter)+"\n"

		if iter%5 == 0:
			pickle.dump(Factors,open(os.path.join(Logs,"Factors_"+str(iter)),"wb"))
			pickle.dump(G,open(os.path.join(Logs,"Core_"+str(iter)),"wb"))

		start = datetime.datetime.now()

		FactorsT_T = [np.dot(M.T,M) for M in Factors]


		G_1 = dtensor(G[0])
		G_2 = dtensor(G[1])
		G_3 = dtensor(G[2])
		
		## Updating A

		GB = G_1.ttm(Factors[1], mode =1, transp=False)
		GB = GB.unfold(0)		
		GC = G_3.ttm(Factors[2], mode =1, transp=False)
		GC = GC.unfold(0)
		Num_A = X1.unfold(0,transp=False).tocsr()
		Num_A = Num_A.dot(GB.T)
		Num_A += X3.unfold(0,transp=False).tocsr().dot(GC.T)
		Denom_A = np.dot(GB, GB.T) + np.dot(GC, GC.T)
		Denom_A = np.dot(Factors[0], Denom_A)
		Denom_A += np.multiply(lambda_a, Factors[0])
		Denom_A += ep

		## updating B

		GA = G_1.ttm(Factors[0], mode =0, transp=False)
		GA = GA.unfold(1)
		GC = G_2.ttm(Factors[2], mode =1, transp=False)
		GC = GC.unfold(0)
		Num_B = X1.unfold(1,transp=False).tocsr()
		Num_B = Num_B.dot(GA.T)
		Num_B += X2.unfold(0,transp=False).tocsr().dot(GC.T)
		Denom_B = np.dot(GA, GA.T) + np.dot(GC, GC.T)  
		Denom_B = np.dot(Factors[1],Denom_B)
		Denom_B += np.multiply(lambda_b, Factors[1])
		Denom_B += ep

		## updating C

		GB = G_2.ttm(Factors[1], mode =0, transp=False)
		GB = GB.unfold(1)
		GA = G_3.ttm(Factors[0], mode =0, transp=False)
		GA = GA.unfold(1)
		Num_C = X2.unfold(1,transp=False).tocsr()
		Num_C = Num_C.dot(GB.T)
		Num_C += X3.unfold(1,transp=False).tocsr().dot(GA.T)
		Denom_C = np.dot(GA, GA.T) + np.dot(GB, GB.T)
		Denom_C = np.dot(Factors[2],Denom_C)
		Denom_C += np.multiply(lambda_c, Factors[2])
		Denom_C += ep

		## Updating Cores 

		Num = X1.ttm(Factors[0],mode = 0,transp=True)
		Num = Num.ttm(Factors[1], mode=1, transp=True)
		Denom = G_1.ttm(FactorsT_T[0],mode=0,transp=True)
		Denom = Denom.ttm(FactorsT_T[1], mode=1, transp=True)
		Denom += ep

		G[0] = np.multiply(G[0], np.divide(Num, Denom))


		Num = X2.ttm(Factors[1],mode = 0,transp=True)
		Num = Num.ttm(Factors[2], mode=1, transp=True)
		Denom = G_2.ttm(FactorsT_T[1],mode=0,transp=True)
		Denom = Denom.ttm(FactorsT_T[2], mode=1, transp=True)
		Denom += ep

		G[1] = np.multiply(G[1], np.divide(Num, Denom))

		Num = X3.ttm(Factors[0],mode = 0,transp=True)
		Num = Num.ttm(Factors[2], mode=1, transp=True)
		Denom = G_3.ttm(FactorsT_T[0],mode=0,transp=True)
		Denom = Denom.ttm(FactorsT_T[2], mode=1, transp=True)
		Denom += ep

		G[2] = np.multiply(G[2], np.divide(Num, Denom))


		Factors[0] = np.multiply(Factors[0], np.divide(Num_A, Denom_A))
		Factors[1] = np.multiply(Factors[1], np.divide(Num_B, Denom_B))
		Factors[2] = np.multiply(Factors[2], np.divide(Num_C, Denom_C))


		'''if FIT =='Y':

			G_1 = dtensor(G[0])
			G_2 = dtensor(G[1])
			G_3 = dtensor(G[2])

			
			normRes1 = sqrt(np.abs(normX1 ** 2 - norm(G[0]) ** 2))
			normRes2 = sqrt(np.abs(normX2 ** 2 - norm(G[1]) ** 2))
			normRes3 = sqrt(np.abs(normX3 ** 2 - norm(G[2]) ** 2))

			fit[0] = 1 - (normRes1/normX1)
			fit[1] = 1 - (normRes2/normX2)
			fit[2] = 1 - (normRes3/normX3)

			avgFit = sum(fit)/3

			print "\n Average Fit: "+str(avgFit)
			print "\n Fit: "+str(fit)

			if abs(avgFit-avgFitOld) < conv:
				break'''

		end = datetime.datetime.now()

		print "Time taken for Iteration: "+str(end-start)



	end = datetime.datetime.now()
	total_time = str(end-gstart)
	print "\n Total factorization time: "+total_time

	pickle.dump(Factors,open(os.path.join(odir,"Factors"),"wb"))
	pickle.dump(G,open(os.path.join(odir,"Core"),"wb"))



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

	fp = open(os.path.join(odir,'Agents.txt'), 'w')
	for col in range(rank[0]):
		topIds = np.argsort(Factors[0][:,col])[::-1][:tops]
		fp.write("Topic-"+str(col)+":")
		l = []
		for id in topIds:
			fp.write("\t"+agents[id]+" ,"+str(Factors[0][id,col]))
			l.append(agents[id])

		Agents.append(l)

		fp.write("\n")

	fp.close()

	fp = open(os.path.join(odir,'Patients.txt'), 'w')
	for col in range(rank[1]):
		topIds = np.argsort(Factors[1][:,col])[::-1][:tops]
		fp.write("Topic-"+str(col)+":")
		l = []
		for id in topIds:
			fp.write("\t"+patnts[id]+" ,"+str(Factors[1][id,col]))
			l.append(patnts[id])

		Patients.append(l)
		fp.write("\n")

	fp.close()

	fp = open(os.path.join(odir,'Instruments.txt'), 'w')
	for col in range(rank[2]):
		topIds = np.argsort(Factors[2][:,col])[::-1][:tops]
		fp.write("Topic-"+str(col)+":")
		l = []
		for id in topIds:
			fp.write("\t"+instmnts[id]+" ,"+str(Factors[2][id,col]))
			l.append(instmnts[id])

		Instruments.append(l)

		fp.write("\n")

	fp.close()

	
	sp = open(os.path.join(odir, "Schemas.txt"),'w')
	for p in range(len(predicates)):
		sp.write(predicates[p]+":\n")
		ids1 = np.argsort(G[0][:,:,p], axis=None)[::-1][:8]
		ids2 = np.argsort(G[1][:,:,p], axis=None)[::-1][:8]
		ids3 = np.argsort(G[2][:,:,p], axis=None)[::-1][:8]

		idx1 = [dictIDS1[idx] for idx in ids1]
		idx2 = [dictIDS2[idx] for idx in ids2]
		idx3 = [dictIDS3[idx] for idx in ids3]

		schemas = getSchema(idx1[:3],idx2[:3],idx3[:3])

		if len(schemas) == 0:
			schemas = getSchema(idx1,idx2,idx3)

		h_schemas = mergeSchemas(schemas)
		if len(h_schemas)>0:
			for s in range(len(h_schemas)):
				schm = h_schemas[s]
				print schm
				sp.write(str(Agents[schm[0]])+"\n")
				sp.write(str(Patients[schm[1]])+"\n")
				sp.write(str(Instruments[schm[2]])+"\n")
				sp.write(str(Instruments[schm[3]])+"\n")
				sp.write("---------------------\n")


		if len(schemas)>0:
			for s  in range(len(schemas)):
				schm = schemas[s]
				print schm
				sp.write(str(Agents[schm[0]])+"\n")
				sp.write(str(Patients[schm[1]])+"\n")
				sp.write(str(Instruments[schm[2]])+"\n")
				sp.write("---------------------\n")

		sp.write("\n\n")

	sp.close()


	
	for g in range(len(G)):
		np_file = os.path.join(odir, "Schemas_"+str(g)+".txt")
		fp = open(np_file,'w')
		for p in range(len(predicates)):			
			ids = np.argsort(G[g][:,:,p], axis=None)[::-1][:5]
			fp.write(predicates[p]+":\n")
			fp.write(dictIDS[g][ids[0]]+"\t"+dictIDS[g][ids[1]]+"\t"+dictIDS[g][ids[2]]+"\t"+dictIDS[g][ids[3]]+"\t"+dictIDS[g][ids[4]]+"\n")

		fp.close()


	
	start = datetime.datetime.now()
	## Fit Computation
	
	if FIT == 'Y':
		print "Computing Fit ...\n"
		f = 0
		A = np.array(X1.subs)
		for p in range(len(predicates)):			
			sidx = np.where(A[2]==p)
			G_p = np.dot(Factors[0], np.dot(G[0][:,:,p], Factors[1].T))
			for idx in sidx:
				G_p[A[0,idx],A[1,idx]] = X1.vals[idx] - G_p[A[0,idx],A[1,idx]]
			
			
			f += np.linalg.norm(G_p)

		fit[0] = 1 - (f/normX1)


		f = 0
		A = np.array(X2.subs)		
		for p in range(len(predicates)):			
			sidx = np.where(A[2]==p)
			G_p = np.dot(Factors[1], np.dot(G[1][:,:,p], Factors[2].T))
			for idx in sidx:
				G_p[A[0,idx],A[1,idx]] = X2.vals[idx] - G_p[A[0,idx],A[1,idx]]
			
			
			f += np.linalg.norm(G_p)

		fit[1] = 1 - (f/normX2)

		f = 0		
		A = np.array(X3.subs)		
		for p in range(len(predicates)):			
			sidx = np.where(A[2]==p)
			G_p = np.dot(Factors[0], np.dot(G[2][:,:,p], Factors[2].T))
			for idx in sidx:
				G_p[A[0,idx],A[1,idx]] = X3.vals[idx] - G_p[A[0,idx],A[1,idx]]
			
			
			f += np.linalg.norm(G_p)

		fit[2] = 1 - (f/normX3)
		
				
		avgFit = sum(fit)/3

		fp = open(os.path.join(odir,"Fit.txt"),'w')
		fp.write("Fit:\t")
		fp.write(str(fit)+"\n")
		fp.write("Average Fit: "+str(avgFit))
		fp.close()

		end = datetime.datetime.now()
		print "Fit Computation Time: "+str(end-start)

def getLambdas(minLambda, maxLambda, step):

	minLmda_a = minLambda[0]
	
	lambdas = []

	while minLmda_a<=maxLambda[0]-step:
		minLmda_b = minLambda[1]
		while minLmda_b<= maxLambda[1]-step:
			minLmda_c = minLambda[2]
			while minLmda_c<= maxLambda[2]-step:
				l =[]
				l.append(minLmda_a)
				l.append(minLmda_b)
				l.append(minLmda_c)
				lambdas.append(l)
				minLmda_c += step

			minLmda_b += step
		minLmda_a += step

	l = [maxLambda[0],maxLambda[1],maxLambda[2]]
	lambdas.append(l)

	return lambdas





if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Read the command line arguments

	parser.add_argument("data", help="Enter the path to data dir", type=str)
	parser.add_argument("odir", help="Enter the path to Output Folder", type=str)
	parser.add_argument("--minLambda", help="Enter the min lambda (list), default = 0.1 0.1 0.1", nargs = '+',type=float)
	parser.add_argument("--maxLambda", help="Enter the max lambda (list), needed only for grid search",nargs = '+', type=float)
	parser.add_argument("--step", help="Enter the step size for grid search (default = 0.5)", type=float, default=0.5)
	#parser.add_argument("K", help="Enter the rank of factorization", type=int)
	#parser.add_argument("--lambda_a", help="Enter the Regularization coefficient for A (default = 0.01)", type=float, default=0.01)
	#parser.add_argument("--lambda_b", help="Enter the Regularization coefficient for B (default = 0.01)", type=float, default=0.01)
	#parser.add_argument("--lambda_c", help="Enter the Regularization coefficient for C (default = 0.01)", type=float, default=0.01)
	parser.add_argument("--maxIters", help="Enter the maximum iterations (default = 10)", type=int, default=10)
	parser.add_argument("--rank1", help="Enter rank1 (default = 10)", type = int, default =10)
	parser.add_argument("--rank2", help="Enter rank2 (default = 10)", type = int, default =10)
	parser.add_argument("--rank3", help="Enter rank3 (default = 10)", type = int, default =10)
	parser.add_argument("--fit", help="Y/N, default = N", type=str, default='N')
	parser.add_argument("--cores", help="Number of Threads", type=int, default=15)
	
	
	args = parser.parse_args()
	rank = [args.rank1, args.rank2, args.rank3]
	Iters = args.maxIters

	
	start = datetime.datetime.now()
	#X1, X2, X3, agents, patnts, instmnts, predicates = loadData(args.data)
	loadData(args.data)
	end = datetime.datetime.now()
	print "Data Loading done: "+str(end-start)+"hrs"


	if not os.path.exists(args.odir):
		os.makedirs(args.odir)

	ncores = args.cores

	if args.maxLambda:
		lambdas = getLambdas(args.minLambda, args.maxLambda, args.step)
		print "Total Combinations: "+str(len(lambdas))
		Parallel(n_jobs=ncores, verbose=1)(delayed(NN_Factorize)(rank, Iters, args.odir, lambdas[i], args.fit) for i in range(len(lambdas)))


	else:
		lambdas = args.minLambda
		NN_Factorize(rank, Iters, args.odir, lambdas, args.fit)





