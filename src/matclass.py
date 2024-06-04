import numpy as np
import math
from line_profiler import profile

#1.1/1.2
class Matrix:
    @profile
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = np.random.random((rows, cols))
    def __str__ (self):
        return str(self.matrix)
    def square_matrix (self):
        return self.rows == self.cols
    # a method that converts a matrix object from the matrix class into a different format(float by default)
    def mattype(self, dtype="float"):
        return self.matrix.astype(dtype)
    def transpose (self):
        self.matrix = np.transpose(self.matrix)
        self.rows, self.cols = self.cols, self.rows
    def element (self, i, j):
        return self.matrix[i-1][j-1]
    def eye (self, n):    #pour voir la matrice identité, not sure that it's good here 
        self.matrix = np.eye(n)
    def shape(self):
        return self.matrix.shape
    def is_square(self):  #+++++++++++++++++
        return self.rows == self.cols  #++++++++++++++++++++++++++++++++++
    def diagonalizable (self):
        eigenvalues = np.linalg.eigvals(self.matrix)   #np.allclose verify if all eigen values are distinctes
        return np.allclose(eigenvalues, np.unique(eigenvalues)) #np.unique is using to obtain the unique eigen values 
    #la fonction renverra true or false
    @profile
    def to_binary_matrix(self, threshold):
        return np.where(self.matrix > threshold, 1, 0)
    
class DenseMatrix(Matrix):
    def add(self, other):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result
    @profile
    def multiply_vector(self, vector):
        result = [0] * self.rows
        for i in range(self.rows):
            for j in range(self.cols):
                result[i] += self.matrix[i][j] * vector[j]
        return result
    def multiplication(self, other):
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return result
    def subtraction(self, other):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return result
    def norm_L1(self):
        column_sums = [0] * self.cols 
        for j in range(self.cols):  # Itérer sur chaque colonne
            for i in range(self.rows):  # Itérer sur chaque ligne
                column_sums[j] += abs(self.matrix[i][j])  # Ajouter la valeur absolue de l'élément à la somme de la colonne
        return max(column_sums) 
    @profile 
    def norm_L2(self):
        # Calculer la somme des carrés de tous les éléments de la matrice
        sum_of_squares = 0
        for i in range(self.rows):
            for j in range(self.cols):
                sum_of_squares += self.matrix[i][j] ** 2
        # Calculer la racine carrée de la somme des carrés pour obtenir la norme L2
        return math.sqrt(sum_of_squares)
    def norm_Linf(self):
        # Calculer la somme des valeurs absolues de chaque ligne
        row_sums = [0] * self.rows  
        for i in range(self.rows):  # Itérer sur chaque ligne
            for j in range(self.cols):  # Itérer sur chaque colonne
                row_sums[i] += abs(self.matrix[i][j])  # Ajouter la valeur absolue de l'élément à la somme de la ligne
        return max(row_sums)  

class SparseMatrix(Matrix):
    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.sparse_representation = self._convert_to_sparse(matrix)
    def _convert_to_sparse(self, matrix):
        sparse_representation = []
        for i in range(self.rows):
            for j in range(self.cols):
                if matrix[i][j] != 0:
                    sparse_representation.append((i, j, matrix[i][j]))
        return sparse_representation
    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Les matrices doivent avoir les mêmes dimensions pour l'addition.")
        result = []
        for (i, j, val) in self.sparse_representation:
            result.append((i, j, val))

        for (i, j, val) in other.sparse_representation:
            result.append((i, j, val))
        return result
    def sub(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Les matrices doivent avoir les mêmes dimensions pour la soustraction.")
        result = []
        for (i, j, val) in self.sparse_representation:
            result.append((i, j, val))
        for (i, j, val) in other.sparse_representation:
            result.append((i, j, -val))
        return result
    def matrix_x_vector(self, vector):
        if len(vector) != self.cols:
            raise ValueError("Le vecteur doit avoir la même longueur que le nombre de colonnes de la matrice.")
        result_vector = [0] * self.rows
        for (i, j, val) in self.sparse_representation:
            result_vector[i] += val * vector[j]
        return result_vector
    def matrix_x_matrix(self, other):
        if self.cols != other.rows:
            raise ValueError("Le nombre de colonnes de la première matrice doit être égal au nombre de lignes de la seconde.")
        result = {}
        for (i, j, val1) in self.sparse_representation:
            for (k, l, val2) in other.sparse_representation:
                if j == k:  
                    if (i, l) not in result:
                        result[(i, l)] = 0
                    result[(i, l)] += val1 * val2
        result_list = []
        for (i, j), val in result.items():
            if val != 0:
                result_list.append((i, j, val))
        return result_list
    def norm_L1(self):
        norm = 0
        for (_, _, val) in self.sparse_representation:
            norm += abs(val)
        return norm
    def norm_L2(self):
        sum_of_squares = 0
        for (_, _, val) in self.sparse_representation:
            sum_of_squares += val ** 2
        return math.sqrt(sum_of_squares)
    def norm_Linf(self):
        max_val = 0
        for (_, _, val) in self.sparse_representation:
            if abs(val) > max_val:
                max_val = abs(val)
        return max_val