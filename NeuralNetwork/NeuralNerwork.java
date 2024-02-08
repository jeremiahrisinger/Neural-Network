package NeuralNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeuralNerwork extends BaseObject {
    private static final Logger<NeuralNerwork> LOGGER = new Logger<NeuralNerwork>(NeuralNerwork.class);

    public static final String INPUT_HIDDEN = "INPUT_HIDDEN";
    public static final String HIDDEN_OUTPUT = "HIDDEN_OUTPUT";
    public static final String HIDDEN_BIAS = "HIDDEN_BIAS";
    public static final String OUTPUT_BIAS = "OUTPUT_BIAS";

    Map<String, Matrix> matMap = new HashMap<>();
    double learningRate = 0.01;

    public NeuralNerwork(int i, int h, int o) {
        matMap.put(INPUT_HIDDEN, new Matrix(h, i));
        matMap.put(HIDDEN_OUTPUT, new Matrix(o, h));
        matMap.put(HIDDEN_BIAS, new Matrix(h, 1));
        matMap.put(OUTPUT_BIAS, new Matrix(o, 1));
    }

    /**
     * This method will take the input for the {@link NeuralNerwork} and calculate
     * each layer of the network to make a predicion
     * 
     * @param x
     * @return
     * @throws Exception
     */
    public List<Double> predict(double[] x) throws Exception {
        LOGGER.logMethod("predict");

        Matrix input = Matrix.fromArray(x);

        Matrix hidden = calculateLayer(input, INPUT_HIDDEN, HIDDEN_BIAS);

        Matrix output = calculateLayer(hidden, HIDDEN_OUTPUT, OUTPUT_BIAS);

        return output.toArray();
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

        Matrix input = Matrix.fromArray(X);

        Matrix hidden = calculateLayer(input, INPUT_HIDDEN, HIDDEN_BIAS);

        Matrix output = calculateLayer(hidden, HIDDEN_OUTPUT, OUTPUT_BIAS);

        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.subtract(target, output);

        calculateBackPropigation(hidden, output, error, HIDDEN_OUTPUT, OUTPUT_BIAS);

        Matrix weightTarget = Matrix.transpose(matMap.get(HIDDEN_OUTPUT));
        Matrix hidden_errors = Matrix.dotProduct(weightTarget, error);

        calculateBackPropigation(input, hidden, hidden_errors, INPUT_HIDDEN, HIDDEN_BIAS);

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
    private Matrix calculateLayer(Matrix prevLayer, String weightName, String biasName) throws Exception {
        LOGGER.logMethod("calculateLayer");
        LOGGER.debug(biasName);
        Matrix curentLayer = Matrix.dotProduct(matMap.get(weightName), prevLayer);
        curentLayer.add(matMap.get(biasName));
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
    private void calculateBackPropigation(Matrix currLayer, Matrix prevLayer, Matrix error, String weightName,
            String biasName) throws Exception {
        LOGGER.logMethod("calculateBackPropigation");

        Matrix gradient = prevLayer.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(learningRate);

        Matrix who_delta = Matrix.dotProduct(gradient, Matrix.transpose(currLayer));

        matMap.get(weightName).add(who_delta);
        matMap.get(biasName).add(gradient);
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
}
