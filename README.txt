Certain functions are stored in a separate module in another file, "recommendation_algorithm.py"
The matrix class and its methods are stored in "matclass.py" file
They are imported into main.py, so that the code can be executed

When using the main module, the terminal commands go like this: python3 main.py <choice> <matrix_filename>
<choice> is a number between 1 and 6. Each number corresponds to a particular part of code to execute.
<matrix_filename> is the path of the matrix, which will be used in part 2's task. However, if instead of the file name, the user writes 'none', then the script will use a randomly-generated matrix instead.
 1 check how much time it takes to calculate eigenvalues for a randomly-generated matrix
 2 compare different matrix-solving algorithms based on computing time
 3 test the randomly-generated matrices and convert them into binary format.
 4 display the graph of singular values
 5 show the matrix and show the number of non-zero rows
 6 execute the recommendation algorithm. 


1. Want to check how much time it takes to calculate eigenvalues for a randomly-generated matrix?
Input five more arguments: <startN>, <endN>, <startM>, <endM>, <filename>
<startM> and <startN> are for the starting matrix's number of rows and columns respectively. 
<endM> and <endN> are the maximum possible values for the matrix's rows/columns. 
<filename> is the filepath for the graph to be saved.
if these arguments aren't input, then the function will use default values and save the graph in the same folder 
as main.py under the name "image.png"

2. Want to compare different matrix-solving algorithms based on computing time?
Input one more argument, <filename>. It is the filepath for the graph to be saved.
If it's not input, then the function will save the graph in the same folder as main.py under the name "image.png"

3. If you want to test two random matrices of a certain size and convert them into binary matrices, no need to input any more arguments.

4. If you want to display the graph of singular values, you can input one more argument: <graph_filename>
<graph_filename> is the path of the file for the graph. If you don't declare it, 
the graph will be saved in the same folder as the main script, under the name "image.png"

5. If you want to test how many non-zero columns are the resulting v_transpose matrix, then no additional arguments needed.

6. If you want to execute the recommendation algorithm, you input two more arguments: <liked_movie_index> <select_movie>.
If you don't input them, then <liked_movie_index> will be 0, and <select_movie> will be 1. 
You will also get a warning about using the default values.
<liked_movie_index> is the number of the movie that the user liked. 
<select_movie> is the number of movies, whose 'liking' values need to be displayed. The liking values are between 0 and 1.
Both <liked_movie_index> and <select_movie> arguments should be integers.