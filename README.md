# TFBA

Codes for "Higher Order Relation Schema Induction".

## Prerequisites:
install sktensor (https://github.com/mnick/scikit-tensor)

This package contains the following files:

* dataGen.py -- Used to generate tensors from the set of tuples.
* factorize.py -- Joint tensor factorization.
* cliqueMine.py -- Constrained Clique mining.


## Running Instructions:

* python2.7 dataGen.py <tuples_file> <output_dir> </br>
	--- Each line in the input file is a tab separated 4-tuple of the format 
		subject "\t" relation "\t" object "\t" other "\t" frequency. </br>
 	--- 3-tuples can also be provided in the same file along with 4-tuples, in which case use the string "<na>" for other. </br>
	--- This script will create pkl files in the output directory. </br>

* python2.7 factorize.py <data_dir> <output_dir> [other options]</br>
	--- Performs the factorization and store the latent factor matrices and core tensors in the <output_dir> directory.
	--- <data_dir> should be same as the <output_dir> of dataGen.py.
	optional arguments:
		  -h, --help            show this help message and exit
		  --minLambda MINLAMBDA [MINLAMBDA ...]
				        Enter the min lambda (list), default = 0.1 0.1 0.1
		  --maxLambda MAXLAMBDA [MAXLAMBDA ...]
				        Enter the max lambda (list), needed only for grid
				        search. If no grid search, provide only minLambda option.
		  --step STEP           Enter the step size for grid search (default = 0.5)
		  --maxIters MAXITERS   Enter the maximum iterations (default = 10)
		  --rank1 RANK1         Enter rank1 (default = 10)
		  --rank2 RANK2         Enter rank2 (default = 10)
		  --rank3 RANK3         Enter rank3 (default = 10)
		  --fit FIT             Y/N, default = N. Give Y for fit computation. 
		  --cores CORES         Number of Threads


* python2.7 cliqueMine.py <data_dir> <output_dir> --rank r1 r2 r3 </br>
	--- Performs constrained clique mining and stores the schemas in <output_dir>
	--- <data_dir> should be same as <data_dir> used to run Factorize.py
		
