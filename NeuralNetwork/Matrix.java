package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * <h3>Matrix Class (extends BaseObject)</h3>
 * <p>
 * This is a class to hold, compare, and manipulate a double[][] matricies. This
 * class was built to be used in a Neural Network.
 * </p>
 * 
 */
public class Matrix extends BaseObject {
    private static final Logger<Matrix> LOGGER = new Logger<Matrix>(Matrix.class);

    private double[][] data;
    int columns;
    int rows;

    public Matrix(int rows, int columns) {
        assert (rows > 0 && columns > 0) : "The rows and columns must be greater than 0.";

        data = new double[rows][columns];
        this.rows = rows;
        this.columns = columns;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                data[r][c] = Math.random() * 2 - 1;
            }
        }
    }

    /**
     * This method will add [scaler] to each data point in the matrix
     * 
     * @param scaler
     */
    public void add(double scaler) {
        LOGGER.logMethod("add");

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                this.data[r][c] += scaler;
            }
        }
    }

    /**
     * This method will add matrix m to this matrix where each row column
     * pair will be added together.
     * 
     * @param m
     * @throws Exception
     */
    public void add(Matrix m) throws Exception {
        LOGGER.logMethod("add");

        if (!this.isSameSize(m)) {
            throw new Exception("Matricies are not the same size.");
        }

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                this.data[r][c] += m.data[r][c];
            }
        }
    }

    /**
     * This method will return true if the rows and columns are equal
     * size
     * 
     * @param m
     * @return
     */
    private boolean isSameSize(Matrix m) {
        LOGGER.logMethod("isSameSize");

        return this.columns == m.columns && this.rows == m.rows;
    }

    /**
     * Static method that will return a matrix where each point
     * ret[r][c] equals a[r][c] - b[r][c]
     * 
     * @param a
     * @param b
     * @return A new Matrix
     * @throws Exception
     */
    public static Matrix subtract(Matrix a, Matrix b) throws Exception {
        LOGGER.logMethod("subtract");

        if (!a.isSameSize(b)) {
            throw new Exception("Matricies are not the same size.");
        }
        Matrix ret = new Matrix(a.rows, a.columns);

        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.columns; c++) {
                ret.data[r][c] = a.data[r][c] - b.data[r][c];
            }
        }

        return ret;
    }

    /**
     * Static method that will return a new Matrix where each point
     * matrix[r][c] is moved to matrix[c][r].
     * 
     * @param a
     * @return new Matrix[c][r]
     */
    public static Matrix transpose(Matrix a) {
        LOGGER.logMethod("transpose");

        Matrix ret = new Matrix(a.columns, a.rows);
        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.columns; c++) {
                ret.data[c][r] = a.data[r][c];
            }
        }

        return ret;
    }

    /**
     * Static method that will compute the dot product of 2 matricies.
     * The resulting matrix will have a.rows number of rows and b.columns
     * number of columns.
     * 
     * @param a
     * @param b
     * @return
     */
    public static Matrix dotProduct(Matrix a, Matrix b) {
        LOGGER.logMethod("dotProduct");

        Matrix temp = new Matrix(a.rows, b.columns);
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.columns; j++) {
                double sum = 0;
                for (int k = 0; k < a.columns; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                temp.data[i][j] = sum;
            }
        }

        return temp;
    }

    /**
     * This method will multiply each value[r][c] in the parameter to
     * each corresponding value[r][c] in this.data.
     * 
     * @param a
     */
    public void multiply(Matrix a) {
        LOGGER.logMethod("multiply");

        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.columns; c++) {
                this.data[r][c] *= a.data[r][c];
            }
        }
    }

    /**
     * This is a direct scaling of the matrix. Each value[r][c] will be
     * multiplied by the scalar (a).
     * 
     * @param a
     */
    public void multiply(double a) {
        LOGGER.logMethod("multiply");

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                this.data[r][c] *= a;
            }
        }
    }

    /**
     * This is will convert each value[r][c] into a number between 1
     * and 0.
     */
    public void sigmoid() {
        LOGGER.logMethod("sigmoid");

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                this.data[r][c] = 1 / (1 + Math.exp(-this.data[r][c]));
            }
        }
    }

    /**
     * This will return a matrix of the derivative of {@link #sigmoid()}
     * 
     * @return
     */
    public Matrix dsigmoid() {
        LOGGER.logMethod("dsigmoid");

        Matrix temp = new Matrix(rows, columns);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++)
                temp.data[r][c] = this.data[r][c] * (1 - this.data[r][c]);
        }
        return temp;
    }

    /**
     * This is a helper method to build start our matrix from an array.
     * 
     * @param x
     * @return
     */
    public static Matrix fromArray(double[] x) {
        LOGGER.logMethod("fromArray");

        Matrix temp = new Matrix(x.length, 1);
        for (int i = 0; i < x.length; i++) {
            temp.data[i][0] = x[i];
        }
        return temp;

    }

    /**
     * This is a helper method to convert our matrix into a List<Double>.
     * 
     * @return
     */
    public List<Double> toArray() {
        LOGGER.logMethod("toArray");

        List<Double> temp = new ArrayList<Double>();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                temp.add(data[i][j]);
            }
        }
        return temp;
    }

    /**
     * to String method for Matrix class. This will build a string in this format:
     * 
     * <pre>
     *  {
     *      1 [a, b, c, d,...]
     *      2 [h, i,...]
     *      ...
     *      10 [...]
     *  }
     * </pre>
     */
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("{\n");

        for (int r = 0; r < data.length; r++) {
            double[] ds = data[r];
            builder.append("\t").append(r).append(" [");
            for (int i = 0; i < ds.length - 1; i++) {
                builder.append(ds[i])
                        .append(", ");
            }
            builder.append(ds[ds.length - 1]).append("]\n");
        }
        builder.append("}");
        return builder.toString();
    }
}
