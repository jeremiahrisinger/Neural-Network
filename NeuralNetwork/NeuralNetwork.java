package NeuralNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * <h3>Neural Network</h3>
 * <p>
 * This class contains a 10 piece map of nodes, error, weights, and biases.
 * </p>
 * Nodes
 * <ul>
 * <li>Input
 * <li>Output
 * <li>Hidden
 * </ul>
 * Weights
 * <ul>
 * <li>Input to Hidden
 * <li>Hidden to Output
 * </ul>
 * Biases
 * <ul>
 * <li>Input to Hidden
 * <li>Hidden to Output
 * </ul>
 * Errors
 * <ul>
 * <li>Error
 * <li>Hidden Error
 * <li>Target
 * </ul>
 */
public class NeuralNetwork extends BaseObject {

    private static final Logger<NeuralNetwork> LOGGER = new Logger<NeuralNetwork>(NeuralNetwork.class);

    private static final String INPUT = "INPUT";
    private static final String OUTPUT = "OUTPUT";
    private static final String HIDDEN = "HIDDEN";

    private static final String INPUT_HIDDEN = "INPUT_HIDDEN";
    private static final String HIDDEN_OUTPUT = "HIDDEN_OUTPUT";

    private static final String HIDDEN_BIAS = "HIDDEN_BIAS";
    private static final String OUTPUT_BIAS = "OUTPUT_BIAS";

    private static final String TARGET = "TARGET";
    private static final String ERROR = "ERROR";
    private static final String HIDDEN_ERROR = "HIDDEN_ERROR";

    Map<String, Matrix> matMap = new HashMap<>();
    double learningRate = 0.005;

    public NeuralNetwork(int i, int h, int o) {
        put(INPUT_HIDDEN, new Matrix(h, i));
        put(HIDDEN_OUTPUT, new Matrix(o, h));
        put(HIDDEN_BIAS, new Matrix(h, 1));
        put(OUTPUT_BIAS, new Matrix(o, 1));
    }

    /**
     * This method will take the input for the {@link NeuralNetwork} and calculate
     * each layer of the network to make a predicion
     * 
     * @param x
     * @return
     * @throws Exception
     */
    public int predict(double[] x) throws Exception {
        LOGGER.logMethod("predict");

        calculateAllLayers(x);

        return getAnswer(get(OUTPUT).toArray());
    }

    /**
     * This method will run one training cycle on the data in X and Y
     * 
     * @see #calculateLayer(Matrix, String, String)
     * @see #calculateBackPropigation(Matrix, Matrix, Matrix, String, String)
     * @param X
     * @param Y
     * @throws Exception
     */
    public void train(double[] X, double[] Y) throws Exception {
        LOGGER.logMethod("train");

        put(TARGET, Matrix.fromArray(Y));
        calculateAllLayers(X);

        put(ERROR, Matrix.subtract(get(TARGET), get(OUTPUT)));

        calculateBackPropigation(HIDDEN, OUTPUT, ERROR, HIDDEN_OUTPUT, OUTPUT_BIAS);

        put(HIDDEN_ERROR, Matrix.dotProduct(Matrix.transpose(get(HIDDEN_OUTPUT)), get(ERROR)));

        calculateBackPropigation(INPUT, HIDDEN, HIDDEN_ERROR, INPUT_HIDDEN, HIDDEN_BIAS);
    }

    /**
     * This method is used to calculate the values for {@link #INPUT},
     * {@link #HIDDEN}, {@link #OUTPUT}
     * 
     * @param X
     * @throws Exception
     */
    private void calculateAllLayers(double[] X) throws Exception {
        put(INPUT, Matrix.fromArray(X));
        put(HIDDEN, calculateLayer(INPUT, INPUT_HIDDEN, HIDDEN_BIAS));
        put(OUTPUT, calculateLayer(HIDDEN, HIDDEN_OUTPUT, OUTPUT_BIAS));
    }

    /**
     * This method will calculate the current layer based on the previous layer, the
     * weights, and biases
     * associated with these layers
     * 
     * @param prevLayer
     * @param weightName
     * @param biasName
     * @return
     * @throws Exception
     */
    private Matrix calculateLayer(String prevLayer, String weightName, String biasName) throws Exception {
        LOGGER.logMethod("calculateLayer");
        LOGGER.debug(biasName);
        Matrix curentLayer = Matrix.dotProduct(get(weightName), get(prevLayer));
        curentLayer.add(get(biasName));
        curentLayer.sigmoid();
        return curentLayer;
    }

    /**
     * Calculates the back propigation for the previous layer using the error
     * matrix. The error matrix
     * will show what is incorrect and by how much and then this method will adjust
     * each weight and bias
     * by that amount.
     * 
     * @param currLayer
     * @param prevLayer
     * @param error
     * @param weightName
     * @param biasName
     * @throws Exception
     */
    private void calculateBackPropigation(String currLayer, String prevLayer, String error, String weightName,
            String biasName) throws Exception {
        LOGGER.logMethod("calculateBackPropigation");

        Matrix gradient = get(prevLayer).dsigmoid();
        gradient.multiply(get(error));
        gradient.multiply(learningRate);

        Matrix who_delta = Matrix.dotProduct(gradient, Matrix.transpose(get(currLayer)));

        get(weightName).add(who_delta);
        get(biasName).add(gradient);
    }

    /**
     * Overloaded fit method to allow different input type
     * 
     * @see #fit(double[][], double[][], int)
     * @param map
     * @param epochs
     */
    public void fit(Map<String, double[][]> map, int epochs) {
        fit(map.get("INPUT"), map.get("ANSWERS"), epochs);
    }

    /**
     * This method will run 'epochs' number of tranings on all the input data in X
     * and Y
     * 
     * @param X
     * @param Y
     * @param epochs
     */
    public void fit(double[][] X, double[][] Y, int epochs) {
        LOGGER.logMethod("fit");

        for (int i = 0; i < epochs; i++) {
            int sampleN = (int) (Math.random() * X.length);
            LOGGER.debug(String.format("Sample [%s]", sampleN));
            try {
                this.train(X[sampleN], Y[sampleN]);
            } catch (Exception e) {
                LOGGER.error("Error occured while trying to train Neural Network.", e);
            }
        }
    }

    /**
     * This will return the number that is specified by the Neural Network.
     * Translating each item in the array to a 1 or 0 with Math.round() and
     * returning the highest value.
     * 
     * @param output
     * @return
     */
    public static int getAnswer(List<Double> output) {
        int maxIndex = 0;
        Double maxValue = 0.0;
        for (int i = 0; i < output.size(); i++) {
            if (maxValue < output.get(i)) {
                maxValue = output.get(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Accessor put method for private use only.
     * 
     * @param name
     * @param m
     */
    private void put(String name, Matrix m) {
        matMap.put(name, m);
    }

    /**
     * Accessor get method for private use only.
     * 
     * @param name
     * @param m
     */
    private Matrix get(String name) {
        return matMap.get(name);
    }
}
