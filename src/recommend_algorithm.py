import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
import time
import src.matclass as mat

# this function calculates eigenvalues for a randomly-generated matrix and then plots the amount of time needed to do that.
# It's from ex. 1.3
def eigenvalue_timecheck(startMsize, startNsize, maxMsize, maxNsize, filename="image.png"):
	timelist1 = []
	timelist2 = []
	nlist = []
	Nsize = startNsize
	Msize = startMsize

	# a loop that runs for as long as the size of the current matrix doesn't exceed the maximum
	while Nsize < maxNsize and Msize < maxMsize:
		# use the matrix class and its methods to generate a random matrix
		matrix_1 = mat.Matrix(Msize, Nsize)
		matrix_a = matrix_1.mattype()

		# converts a non-square matrix into a square one using singular-value decomposition.
		if startNsize != startMsize:
			s = time.time()
			# one of hte outputs is the Vh matrix, with right singular vectors as rows.
			U, sing, Vh = sp.linalg.svd(matrix_a, full_matrices = True)
			# This is what will be used as the actual matrix, for which to find eigenvalues.
			matrix_a = Vh
			e = time.time()
			ex = e-s
			timelist2.append(ex)

		# computes eigenvalues of the matrix
		s = time.time()
		vals = sp.linalg.eigh(matrix_a, eigvals_only=True)
		e = time.time()
		ex = e-s
		timelist1.append(ex)

		nlist.append(Msize)
		print(Msize, Nsize, sep=", ")
		Nsize *= 2
		Msize *= 2
	# the function that plots the graph of time taken.
	fig, axs = mpl.subplot_mosaic([['linear-log']])
	ax = axs['linear-log']
	ax.plot(nlist, timelist1)
	# if the matrix is not square, then time taken for singular value decomposition is also included
	if len(timelist2)>0:
		ax.plot(nlist, timelist2)
		ax.legend(['Eigenvector finding', 'Singular Value Decomposition'])
	else:
		ax.legend(['Eigenvector finding'])
	ax.set_yscale('log')
	ax.set_xlabel('linear')
	ax.set_ylabel('log')
	ax.set(xlabel='Amount of rows', ylabel='Execution time',
		title='Execution time over matrix size')
	print(timelist1)
	print(nlist)
	fig.savefig(filename)

# function for ex. 1.4
def many_matrix_solver(startsize, maxsize, filename):
	timelist1 = []
	timelist2 = []
	timelist3 = []
	timelist4 = []
	size = startsize
# loop runs while current matrix is smaller than the maximum
	while size < maxsize:
		
		s = time.time()
		# generates the diagonals for the sparse matrix
		diagmain = np.repeat(4, size)
		diag1 = np.repeat(-1, size-1)
		# randomly generates a dense matrix from the matrix class and converts it to a float format
		dense_matrix_1 = mat.Matrix(size, size)
		dense_matrix_a = dense_matrix_1.mattype()
		# randomly generates a sparse matrix. 
		# It's a diagonal matrix, so that its likelihood of being singular is lower
		sparse_matrix_a = sp.sparse.diags([diag1, diagmain, diag1], [1, 0, -1], [size,size])
		# randomly generates a vector
		vector_b = np.random.rand(size)
		e = time.time()
		ex = e - s
		timelist4.append(ex)

		# solve an Ax=b equation, with A being a dense matrix, b being a vector
		# it's allowed to overwrite both A and b, so that it runs faster.
		s = time.time()
		dense_solve = sp.linalg.solve(dense_matrix_a, vector_b, overwrite_a=True, overwrite_b=True)
		e = time.time()
		ex = e - s
		timelist1.append(ex)

		# solve an Ax=b equation, with A being a sparse matrix, b being a vector. Used CSR storage function
		s = time.time()
		csr_matrix_a = sparse_matrix_a.tocsr()
		sparse_solve = sp.sparse.linalg.spsolve(csr_matrix_a, vector_b)
		e = time.time()
		ex = e - s
		timelist2.append(ex)


		# solve an Ax=b equation, with A being a sparse matrix, b being a vector. Used CSC storage function
		s = time.time()
		csc_matrix_a = sparse_matrix_a.tocsc()
		sparse_solve = sp.sparse.linalg.spsolve(csc_matrix_a, vector_b)
		e = time.time()
		ex = e - s
		timelist3.append(ex)

		# double matrix size, until the limit is reached.
		size = size*2

	# the function that plots the graph of time taken.
	fig, axs = mpl.subplot_mosaic([['linear-log']])
	ax = axs['linear-log']
	stepsize = round((maxsize-startsize)/len(timelist1))
	ax.plot(range(startsize, maxsize, stepsize), timelist1)
	ax.plot(range(startsize, maxsize, stepsize), timelist2)
	ax.plot(range(startsize, maxsize, stepsize), timelist3)
	ax.plot(range(startsize, maxsize, stepsize), timelist4)
	ax.legend(['Dense', 'Sparse CSR', 'Sparse CSC', 'Matrix Generation'])
	ax.set_yscale('log')
	ax.set_xlabel('linear')
	ax.set_ylabel('log')
	ax.set(xlabel='Size', ylabel='Execution time',
		title='Execution time over matrix size')
	fig.savefig(filename)



# recommends movies based on the dot-product multiplication. Ex. 2.5
def recommender(V_transpose, liked_movie_index=0, selected_movie_num=1):
		recommended = []
		for i in range(V_transpose.shape[0]):
			if i != int(liked_movie_index):
				recommended.append([i, np.dot(V_transpose[i], V_transpose[int(liked_movie_index)])])
		final_rec = sorted(recommended)
		return final_rec[:int(selected_movie_num)]

# generates a random sparse binary matrix. Useful for testing in part 2
def float_bin_mat(rows, cols, density):
	flat_array=np.random.rand(rows*cols)
	threshold = 1-density
	bin_matrix = (flat_array > threshold).astype(int).reshape(rows, cols)
	float_bin_matrix = bin_matrix.astype(float)
	return float_bin_matrix

def singular_values(rows, cols):
	if min(rows, cols) <= 500:
		k = min(rows, cols)-1
	elif min(rows, cols) < 1000:
		k = (min(rows, cols))/2
	else:
		k = 500
	return k


