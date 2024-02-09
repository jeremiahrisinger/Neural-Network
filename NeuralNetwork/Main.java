package NeuralNetwork;

import java.util.HashMap;

public class Main extends BaseObject {

    private static final int MAX_ADDEND_VALUE = 5;
    private static final int OUTPUT_RANGE = 10;

    private static final int NUM_OUTPUT_NODES = OUTPUT_RANGE + 1;
    private static final int NUM_INPUT_NODES = 2;
    private static final int NUM_HIDDEN_NODES = (NUM_INPUT_NODES * MAX_ADDEND_VALUE
            + NUM_OUTPUT_NODES * MAX_ADDEND_VALUE) / 2;

    private static final int NUMBER_OF_TRAINING_ATTEMPTS = 6000000;

    private static final String ANSWERS = "ANSWERS";
    private static final String INPUT = "INPUT";

    private static final Logger<Main> LOGGER = new Logger<Main>(Main.class);

    private static double[][] inputs = getInputs(MAX_ADDEND_VALUE);
    private static double[][] answers = calculateAnswers(inputs);

    /**
     * Starting point for training and testing the NeuralNetwork
     * 
     * @param args
     */
    public static void main(String[] args) {
        LOGGER.logMethod("main");

        HashMap<String, double[][]> addTraningData = getAdditionInputAndAnswerMap();

        NeuralNetwork nn = new NeuralNetwork(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES);
        nn.fit(addTraningData, NUMBER_OF_TRAINING_ATTEMPTS);

        testNeuralNetwork(nn, 5, MAX_ADDEND_VALUE);
    }

    /**
     * This method will run NeuralNetwork.predict on random inputs a
     * 'numberOOfTests' amount of times and print out the result
     * 
     * @param nn
     * @param numberOfTests
     * @param inputRange
     */
    private static void testNeuralNetwork(NeuralNetwork nn, int numberOfTests, int inputRange) {
        int output = -1;

        for (int i = 0; i < numberOfTests; i++) {
            double[] input = { getRandomInt(inputRange), getRandomInt(inputRange) };
            try {
                output = nn.predict(input);
            } catch (Exception e) {
                LOGGER.error("Error occured while trying to predict output.", e);
            }

            LOGGER.info(String.format("input {%s, %s} -> {%s}\t%s", input[0], input[1], output,
                    output - (input[0] + input[1])));
        }
    }

    /**
     * This method is to fill the data map with input and answer data
     * 
     * @return
     */
    private static HashMap<String, double[][]> getAdditionInputAndAnswerMap() {
        HashMap<String, double[][]> add = new HashMap<>();
        add.put(INPUT, inputs);
        add.put(ANSWERS, answers);
        return add;
    }

    /**
     * This method will create an input array for every combination of numbers
     * between 0 and i
     * 
     * @param i
     * @return
     */
    private static double[][] getInputs(int i) {
        double[][] ret = new double[(i + 1) * (i + 1) + 1][2];
        for (int j = 0; j < i + 1; j++) {
            for (int j2 = 0; j2 < i + 1; j2++) {
                ret[i * j + j2][0] = j;
                ret[i * j + j2][1] = j2;
            }
        }
        return ret;
    }

    /**
     * This method returns a random int between 0 and max
     * 
     * @param max
     * @return
     */
    private static long getRandomInt(int max) {
        return Math.round(Math.random() * max);
    }

    /**
     * This method will calcualate the answer for each input
     * 
     * @param input
     * @return
     */
    private static double[][] calculateAnswers(double[][] input) {
        double[][] answers = new double[inputs.length][11];
        for (int i = 0; i < input.length; i++) {
            int sum = 0;
            for (int j = 0; j < input[i].length; j++) {
                sum += input[i][j];
            }
            answers[i][sum] = 1;
        }
        return answers;
    }

}
