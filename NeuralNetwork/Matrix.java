package NeuralNetwork;

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
}
