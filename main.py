import scipy as sp
import numpy as np
import time
import sys
import matplotlib.pyplot as mpl
import src.recommend_algorithm as rec
import src.matclass as mat

# checks if the matrix used in part 2 will be random
randmatrix = False
# this part checks if the length of the input command is correct. If choice is absent, stop the program. 
# If instead of matrix file name, you write 'none', code uses the random matrix
if len(sys.argv) < 2:
    print("Error! Input the choice value in the terminal and the matrix filepath('none' if none wanted) as well")
    sys.exit(1)
if sys.argv[2] == "none":
    print("You did not specify the file path for your desired matrix." 
        "Therefore, you will use a random matrix for all the tasks in part 2.")
    randmatrix = True

# sets the variables of the sys.argv. Also has a list of acceptable choice values
choice = sys.argv[1]
mat_filename = sys.argv[2]
choicelist = ["1", "2", "3", "4", "5", "6"]

if randmatrix == True:
    # I will use the real binary matrix when it will be available for me.
    # For now, the matrix that I will use is a dummy matrix. It will be converted.
    # later on, I will make sure that the user can input the pathway to the file they want
    # also, there is an alternate way to set the size. 
    # it will be defined using the special matrix class and will be done when Gabriel says it will be done
    rows = 50
    cols = 50

    # returns the randomly-generated matrix. May be replaced with code that lets the user declare whichever file they want.
    float_bin_mat = rec.float_bin_mat(rows, cols, 0.1)
else:
    float_bin_mat = np.loadtxt(mat_filename)
    rows = float_bin_mat.shape[0]
    cols = float_bin_mat.shape[1]

# checks how many singular values need to be saved based on the size of the matrix.
k = rec.singular_values(rows, cols)

# this procedure uses a sparse matrix
# number of highest singular values, depends on the minimum number of rows/columns in the matrix
# matrix Vh1 is V_transpose, which will be important later on
U1, s1, Vh1 = sp.sparse.linalg.svds(float_bin_mat, k = k)


# ex. 1.3. 
if choice == "1":
    # checks if the length of the terminal command is suitable. Otherwise uses default values
    start = time.time()
    if len(sys.argv) > 7:
            startM = int(sys.argv[3])
            endM = int(sys.argv[4])
            startN = int(sys.argv[5])
            endN = int(sys.argv[6])
            filename = sys.argv[7]
            print("Your graph will be saved as ", filename)
    # the default values in question
    else:
        startM = 30
        endM = 100
        startN = 10
        endN = 100
        filename = "image.png"
        print("Insufficient amount of values input. Using default values")
        print("Your graph will be saved as image.png. Any other graph with that name will be overwritten")
    # uses the function in the recommended_algorithm to check how much time it takes 
    #for eigenvalues to be found with different algorithms
    rec.eigenvalue_timecheck(startM, endM, startN, endN, filename)
    end = time.time()
    exec = end-start
    print(startM, endM, startN, endN, sep=" ")
    print("time taken for execution is ", exec, " seconds")

# ex. 1.4.
if choice == "2":
    # checks if the command line is large enough
    if len(sys.argv) > 3:
        filename = sys.argv[3]
        print("Your graph will be saved as ", filename)
    else:
        filename = "image.png"
        print("Your graph will be saved as image.png. Any other graph with that name will be overwritten")
    start = time.time()
    # define matrix size. For now, the size is set in code, but perhaps we will make it possible 
    # for the user to define the matrix size too
    startsize = 500
    maxsize = 5000
    # loop runs while current matrix is smaller than the maximum
    rec.many_matrix_solver(startsize, maxsize, filename)
    end = time.time()
    exec = end-start
    print("time taken for execution is ", exec, " seconds")



# Définition de la fonction lambda pour la conversion en matrice binaire
#binary_matrix = lambda matrix, threshold: np.where(matrix > threshold, 1, 0)

# Examples of test cases. They are a part of task 2.2
# matrix1 = np.array([
#     [2, 0, 3],
#     [0, 4, 0],
#     [1, 0, 5],
#     [6, 0, 7]])

# matrix2 = np.array([
#     [0, 2, 4, 0],
#     [3, 0, 0, 1],
#     [5, 0, 0, 3]])

if choice == "3":
    start = time.time()
    # Définition du seuil pour la conversion en matrice binaire
    threshold = 0.5
    # randomly-generated matrices
    matrix1 = mat.Matrix(3, 3)
    matrix2 = mat.Matrix(4,2)
    # Application de la fonction de conversion en matrice binaire avec le seuil spécifié
    binary_matrix1 = matrix1.to_binary_matrix(threshold)
    binary_matrix2 = matrix2.to_binary_matrix(threshold)
    # lambda function turns a matrix into a binary one
    bin = lambda matrix1, threshold : np.where(matrix1 > threshold, 1, 0)
    # Affichage des résultats
    print("Matrice originale 1:")
    print(matrix1)

    print("\nMatrice binaire 1:")
    print(bin)

    print("\nMatrice originale 2:")
    print(matrix2)

    print("\nMatrice binaire 2:")
    print(binary_matrix2)
    end = time.time()
    exec = end-start
    print("time taken for execution is ", exec, " seconds")

# ex. 2.3.
if choice == "4":
        # if
    if len(sys.argv) > 3:
            filename = sys.argv[3]
    else:
            filename = "image.png"
    # plot the singular values
    start = time.time()
    fig, axs = mpl.subplot_mosaic([['linear-log']])
    ax = axs['linear-log']
    ax.plot(range(1, len(s1)+1), s1)
    ax.legend(['Scipy Sparse'])
    ax.set_yscale('log')
    ax.set_xlabel('linear')
    ax.set_ylabel('log')
    ax.set(xlabel='Order', ylabel='S-value',
            title='Plot of s-values')
    mpl.show()
    fig.savefig(filename)
    end = time.time()
    exec = end-start
    print("time taken for execution is ", exec, " seconds")


# this choice means showing the number of non-zero rows depending on the size of the matrix. Test for task 2.4
if choice == "5":
    start = time.time()
    # create a mask that checks if the columns in matrix Vh3 have all values equal or at least close to 0
    tolerance = 0.05
    zero_columns_mask = np.all(abs(Vh1) <= tolerance, axis=0)
    # NV3 has same values as Vh3, except its dimensions are same as those of the original sparse matrix
    # and all the zero-like columns are removed
    NV3 = Vh1[:, ~zero_columns_mask]
    print(NV3.shape)
    end = time.time()
    exec = end-start
    print("time taken for execution is ", exec, " seconds")

# Ex. 2.5 executes the recommender function from the recommend_algorithm module. 
# As a result, gives a list of indexes of how likely is a movie to get recommended.
if choice == "6":
    # sets default values for the function if the terminal command doesn't have the values
    liked_movie_index = 0
    selected_movies = 1
    # if these values are in the terminal command, then they replace the defaults
    if len(sys.argv) >= 4:
            liked_movie_index = int(sys.argv[3])
    if len(sys.argv) >= 5:
            selected_movies = int(sys.argv[4])
    # warns the user that default values are useds
    else:
            print("The liked_movie_index is 0 and the # of selected movies is 1. \n "
            "If you want different values, please input them as arguments")
    # prints the...
    print(rec.recommender(Vh1, liked_movie_index, selected_movies))

# checks if the choice number is in the list of the accepted commands. If not, then show this message.
if choice not in choicelist:
    print("Invalid choice. Please input the correct choice number")