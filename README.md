# TFBA

Codes for Tensor Factorization with Back-off and Aggregation.

## Prerequisites:
install sktensor (https://github.com/mnick/scikit-tensor)

This package contains the following files:

* dataGen.py -- Used to generate tensors from the set of tuples.
* factorize.py -- Joint tensor factorization.
* cliqueMine.py -- Constrained Clique mining.


## Usage:

$ python2.7 dataGen.py <tuples_file> <output_dir> </br>
	--- Each line in the input file is a tab separated 4-tuple of the format 
		subject "\t" relation "\t" object "\t" other "\t" frequency. </br>
 	--- 3-tuples can also be provided in the same file along with 4-tuples, in which case use the string "<na>" for other. </br>
	--- This script will create pkl files in the output directory. </br>

$ python2.7 factorize.py <data_dir> <output_dir> [other options]</br>
	--- Performs the factorization and store the latent factor matrices and core tensors in the <output_dir> directory. </br>
	--- <data_dir> should be same as the <output_dir> of dataGen.py. </br>
	optional arguments: </br>
		  -h, --help            show this help message and exit </br>
		  --minLambda MINLAMBDA [MINLAMBDA ...] </br>
				        ** Enter the min lambda (list), default = 0.1 0.1 0.1 </br>
		  --maxLambda MAXLAMBDA [MAXLAMBDA ...] </br>
				       ** Enter the max lambda (list), needed only for grid
				        search. If no grid search, provide only minLambda option.
		  --step STEP           Enter the step size for grid search (default = 0.5) </br>
		  --maxIters MAXITERS   Enter the maximum iterations (default = 10) </br>
		  --rank1 RANK1         Enter rank1 (default = 10) </br>
		  --rank2 RANK2         Enter rank2 (default = 10) </br>
		  --rank3 RANK3         Enter rank3 (default = 10) </br>
		  --fit FIT             Y/N, default = N. Give Y for fit computation. </br>
		  --cores CORES         Number of Threads </br>


$ python2.7 cliqueMine.py <data_dir> <output_dir> --rank r1 r2 r3 </br>
	--- Performs constrained clique mining and stores the schemas in <output_dir> </br>
	--- <data_dir> should be same as <data_dir> used to run Factorize.py
	
## References:
[1] Madhav Nimishakavi, Manish Gupta and Partha Talukdar. Relation Schema Induction using Tensor Factorization with Back-off and Aggregation. Proceedings of 2018 Conference on Association for Computaional Linguistics (ACL 2018).
		
